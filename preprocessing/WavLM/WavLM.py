# --------------------------------------------------------
# WavLM: Large-Scale Self-Supervised  Pre-training  for Full Stack Speech Processing (https://arxiv.org/abs/2110.13900.pdf)
# Github source: https://github.com/microsoft/unilm/tree/master/wavlm
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Based on fairseq code bases
# https://github.com/pytorch/fairseq
# --------------------------------------------------------

import math
import logging
from typing import List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from WavLM.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    MultiheadAttention,
    SamePad,
    init_bert_params,
    get_activation_fn,
    TransposeLast,
    GLU_Linear,
)

logger = logging.getLogger(__name__)

##生成掩码索引，根据输入的参数生成适用于给定形状的掩码索引（静态掩码）
def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],   #填充掩码（生成的掩码将与填充掩码合并）
    mask_prob: float,     #掩码概率(控制在哪些位置生成掩码，如果mask_prob=0.1,则表示以0.1的概率在可能的位置生成掩码)
    mask_length: int,     #指定生成的掩码的长度。在生成动态掩码时使用到它
    mask_type: str = "static", #静态掩码，dynamic是动态掩码
    mask_other: float = 0.0, #额外的掩码概率，用于控制生成其他类型的掩码
    min_masks: int = 0,     #生成的最小掩码数量
    no_overlap: bool = False,   #布尔值参数，用于指定是否允许掩码重叠（这里是重叠）
    min_space: int = 0,     #生成的掩码之间的最小间隔
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = np.random.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = np.random.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = np.random.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = np.random.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = np.random.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True

    return mask


class WavLMConfig:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"     # mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with normalize=True)
        self.encoder_layers: int = 12     # num encoder layers in the transformer

        self.encoder_embed_dim: int = 768     # encoder embedding dimension  768
        self.encoder_ffn_embed_dim: int = 3072     # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12     # num encoder attention heads
        self.activation_fn: str = "gelu"     # activation function to use

        self.layer_norm_first: bool = False     # apply layernorm first in the transformer
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"     # string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_bias: bool = False     # include bias in conv encoder
        self.feature_grad_mult: float = 1.0     # multiply feature extractor var grads by this

        self.normalize: bool = False  # normalize input to have 0 mean and unit variance during training

        # dropouts
        self.dropout: float = 0.1     # dropout probability for the transformer
        self.attention_dropout: float = 0.1     # dropout probability for attention weights
        self.activation_dropout: float = 0.0     # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0     # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0     # dropout to apply to the input (after feat extr)
        self.dropout_features: float = 0.0     # dropout to apply to the features (after feat extr)

        # masking
        self.mask_length: int = 10     # mask length
        self.mask_prob: float = 0.65     # probability of replacing a token with mask
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # channel masking
        self.mask_channel_length: int = 10     # length of the mask for features (channels)
        self.mask_channel_prob: float = 0.0     # probability of replacing a feature with 0
        self.mask_channel_selection: str = "static"     # how to choose mask length for channel masking
        self.mask_channel_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
        self.no_mask_channel_overlap: bool = False     # whether to allow channel masks to overlap
        self.mask_channel_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # positional embeddings
        self.conv_pos: int = 128     # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16     # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False     # apply relative position embedding
        self.num_buckets: int = 320     # number of buckets for relative position embedding
        self.max_distance: int = 1280     # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False     # apply gated relative position embedding

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)


class WavLM(nn.Module):
    def __init__(
        self,
        cfg: WavLMConfig,
    ) -> None:
        super().__init__()
        logger.info(f"WavLM Config: {cfg.__dict__}")   #使用日志记录器logger记录了模型配置的信息，以便在运行时查看。

        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]   #从配置中提取最后一个特征提取层的输出大小，存储在self.embed

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, cfg.encoder_embed_dim)
            if self.embed != cfg.encoder_embed_dim
            else None
        )

        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        ) #可学习的张量作为模型的掩码嵌入，其大小由配置中的encoder_embed_dim决定。

        self.encoder = TransformerEncoder(cfg)      #Transformer编码器(encoder)
        self.layer_norm = LayerNorm(self.embed)    #LayerNorm层，用于对特征进行归一化处理
##输入张量x应用掩码，并返回掩码后的张量以及掩码的索引
    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape      #B：批大小；T：序列长度，C：特征通道数
        if self.mask_prob > 0:   #在时间维度上应用掩码
            mask_indices = compute_mask_indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,   #最小掩码数是2
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)   #
            x[mask_indices] = self.mask_emb
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:  #在通道维度上应用掩码
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0    #通过索引操作将生成的掩码应用到输入张量x上，将对应位置的通道置零

        return x, mask_indices
#生成前向注意力机制的填充掩码（返回生成的填充掩码，对填充位置进行屏蔽，确保模型不会关注填充部分的输入）
    def forward_padding_mask(
            self, features: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)     #计算填充掩码的额外长度（填充掩码长度与输入特征长度之间的差异）
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]    #存在额外的填充长度，在填充掩码的末尾删除这些额外的填充部分，以使填充掩码的长度与特征的长度匹配。
        padding_mask = padding_mask.view(
            padding_mask.size(0), features.size(1), -1
        )
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ):

        if self.feature_grad_mult > 0:       #检查是否存在特征梯度乘法器
            features = self.feature_extractor(source)    #使用特征提取器从输入源中提取特征。
            if self.feature_grad_mult != 1.0:   #检查特征梯度乘法器是否不等于1.0
                features = GradMultiply.apply(features, self.feature_grad_mult)  #特征梯度乘法器不等于1.0，则将特征与特征梯度乘法器相乘。
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features = features.transpose(1, 2)   #将特征张量的维度从(B,T,C)转换为(B,C,T)
        features = self.layer_norm(features)  #归一化

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)  #对特征进行输入的dropout操作。

        if mask:   #检查应用掩码器
            x, mask_indices = self.apply_mask(
                features, padding_mask
            )
        else:
            x = features

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool
        ##特征传递给编码器模型进行处理，根据指层次输出
        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )
  #将特征，填充掩码，原始特征和编码器层次结果存储在字典res中。
        res = {"x": x, "padding_mask": padding_mask, "features": features, "layer_results": layer_results}

        feature = res["features"] if ret_conv else res["x"]   #返回特征或卷积结果
        if ret_layer_results:  #检查是否需要返回编码器的层次结果
            feature = (feature, res["layer_results"])
        return feature, res["padding_mask"]  #返回特征、填充掩码

##用于特征提取的卷积模型类
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            conv_type: str = "default"
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}    #mode参数只能是"default"或"layer_norm"
    #一个内部函数block，用于创建卷积块。
        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():    #一个内部函数，用于创建卷积层并初始化权重。
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)  #一个1D卷积层
                nn.init.kaiming_normal_(conv.weight)   #使用Kaiming初始化方法初始化卷积层的权重。
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:           #is_layer_norm为true，则创建包含层归一化的卷积块。
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),   #添加一个丢弃层，用于随机丢弃输入元素以防止过拟合
                    nn.Sequential(
                        TransposeLast(),    #
                        Fp32LayerNorm(dim, elementwise_affine=True),  #应用组归一化操作，其中dim是归一化的维度，使用可学习的缩放和平移参数。
                        TransposeLast(),
                    ),
                    nn.GELU(),   #激活函数
                )
            elif is_group_norm:   #包含组归一化的卷积块
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:   #既不是层归一化，也不是组归一化，则创建普通的卷积块
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
 #构造函数或初始化方法
        self.conv_type = conv_type
        if self.conv_type == "default":     #创建自定义的卷积层
            in_d = 1    #初始化输入维度为1，用于跟踪当前卷积层的输入维度
            self.conv_layers = nn.ModuleList()   #存储所有的卷积层（空的）
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)  #卷积层定义的长度，如果不是，抛出异常并显示错误消息。
                (dim, k, stride) = cl   #将卷积层定义解压为维度、卷积核大小和步长

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.conv_type == "conv2d":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride)  #2D卷积层到conv_layer中，使用输入维度、输出维度、卷积核大小和步长
                )
                self.conv_layers.append(torch.nn.ReLU())  #每个卷积层后添加Relu激活函数
                in_d = dim   #更新输入维度为当前卷积层的输出维度
        elif self.conv_type == "custom":
            in_d = 1
            idim = 80  #最大池化层的参数
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)  #卷积层
                )
                self.conv_layers.append(   #归一化层
                    torch.nn.LayerNorm([dim, idim])
                )
                self.conv_layers.append(torch.nn.ReLU())  #激活函数
                in_d = dim
                if (i + 1) % 2 == 0:  #当前是偶数个卷积层
                    self.conv_layers.append(   #添加一个最大池化层到conv_layers中，使用核大小为2，步长为2的最大池化操作。
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            pass
##前向传播方法
    def forward(self, x, mask=None):

        # BxT -> BxCxT
        x = x.unsqueeze(1)
        if self.conv_type == "custom":
            for conv in self.conv_layers:  #遍历卷积层列表中的每一层
                if isinstance(conv, nn.LayerNorm):  #检查当前卷积层是否是层归一化。
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()   #
            x = x.view(x.size(0), -1, x.size(-1))  #第一维不变，最后一维不变，中间维度根据张量的总元素数量自动计算。
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t) #输出张量x进行维度转置，将时间维度T与特征维度F交换位置，并保证内存连续性。
        return x

##Transformer编码器
class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.pos_conv = nn.Conv1d(     #一维卷积层（处理位置编码）
            self.embedding_dim,
            self.embedding_dim,
            kernel_size=args.conv_pos,
            padding=args.conv_pos // 2,
            groups=args.conv_pos_groups,
        )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.embedding_dim))  #权重初始化的标准差（初始化卷积层的权重）
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)  #正态分布初始化（均值为0，标准差）
        nn.init.constant_(self.pos_conv.bias, 0)  #偏差初始化为0

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)   #权重归一化，确保训练的稳定性
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())  #串联

        if hasattr(args, "relative_position_embedding"):
            self.relative_position_embedding = args.relative_position_embedding
            self.num_buckets = args.num_buckets
            self.max_distance = args.max_distance
        else:
            self.relative_position_embedding = False
            self.num_buckets = 0         #相对位置编码的桶数为0
            self.max_distance = 0        #相对位置编码的最大距离为0

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                    has_relative_attention_bias=(self.relative_position_embedding and i == 0),
                    num_buckets=self.num_buckets,
                    max_distance=self.max_distance,
                    gru_rel_pos=args.gru_rel_pos,
                )
                for i in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, streaming_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, streaming_mask, layer)

        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)

        return x, layer_results

    def extract_features(self, x, padding_mask=None, streaming_mask=None, tgt_layer=None):

        if padding_mask is not None:
            x[padding_mask] = 0

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        layer_results = []
        z = None
        if tgt_layer is not None:
            layer_results.append((x, z))
        r = None
        pos_bias = None
        for i, layer in enumerate(self.layers):
            dropout_probability = np.random.random()
            if not self.training or (dropout_probability > self.layerdrop):
                x, z, pos_bias = layer(x, self_attn_padding_mask=padding_mask, need_weights=False,
                                       self_attn_mask=streaming_mask, pos_bias=pos_bias)
            if tgt_layer is not None:
                layer_results.append((x, z))
            if i == tgt_layer:
                r = x
                break

        if r is not None:
            x = r

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_results


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: float = 768,
            ffn_embedding_dim: float = 3072,
            num_attention_heads: float = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = "relu",
            layer_norm_first: bool = False,
            has_relative_attention_bias: bool = False,
            num_buckets: int = 0,
            max_distance: int = 0,
            rescale_init: bool = False,
            gru_rel_pos: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_name = activation_fn
        self.activation_fn = get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            has_relative_attention_bias=has_relative_attention_bias,
            num_buckets=num_buckets,
            max_distance=max_distance,
            rescale_init=rescale_init,
            gru_rel_pos=gru_rel_pos,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)

        if self.activation_name == "glu":
            self.fc1 = GLU_Linear(self.embedding_dim, ffn_embedding_dim, "swish")
        else:
            self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: torch.Tensor = None,
            self_attn_padding_mask: torch.Tensor = None,
            need_weights: bool = False,
            pos_bias=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn, pos_bias = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
                position_bias=pos_bias
            )

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            if self.activation_name == "glu":
                x = self.fc1(x)
            else:
                x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn, pos_bias

