import os
import pickle
import torch
import librosa
import numpy as np
#加载wavlm模型
from WavLM.WavLM import WavLM,WavLMConfig

##判断是否用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False

checkpoint = torch.load(r'WavLM-Large.pt')#Pretrained model checkpoint path.
cfg = WavLMConfig(checkpoint['cfg'])  #从检查点中加载配置信息。（包括模型的参数配置，例如：层数，隐藏单元数）
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()   #将模型设置为评估模型（模型将不会进行梯度计算或参数更新）

##模型pkl文件的文件夹路径
model_path = "pro_model_label_four_EMODB"
os.makedirs(model_path, exist_ok=True)
len_wav = 48000

##指定包含.pkl文件的文件夹路径
folder_path = r'wav_label_m'

##获取文件夹中的所有文件的列表
file_list = os.listdir(folder_path)
##筛选出.pkl文件
pkl_files = [file for file in file_list if file.endswith('.pkl')]

# pkl_datas = ["model_feature_label_1"]

# 定义一个字典来存储处理结果
result_dict = {}
num_i = 0

##遍历每个.pkl文件并加载
for pkl_file in pkl_files:
    num_i = num_i + 1
    print("num_i:",num_i)
    pkl_file_path = os.path.join(folder_path,pkl_file)
    # print("pkl_path:",pkl_file_path)
    with open(pkl_file_path,'rb') as f:   #加载
        fold_data = pickle.load(f)
    train_indices = fold_data['train']
    test_indices = fold_data['test']
    print(len(train_indices))

    # 定义一个列表来存储当前文件处理后的结果
    current_results_list_train = []
    current_results_list_test = []

    for train_sample in train_indices:
        filename_train,label_train = train_sample
        # print("训练集文件名：",filename_train,"训练集标签：",label_train)
        data_,_ = librosa.load(filename_train,sr=16000)

        data_m = data_
        if len(data_m) <= len_wav:  # 检测长度，是否小于等于32000个样本点
            pad_width = len_wav - len(data_m)
            data_m = np.pad(data_m, (0, pad_width), mode='constant', constant_values=0)  # 末尾补零
            # print("a",len(data_m))

        else:
            data_m = data_m[0:len_wav]
            # print(len(data_m))

        data_w = data_m

        data = data_w[np.newaxis, :]  # 将音频数据转换为pytorch张量，并添加一个新的维度作为批处理维度。
        data = torch.Tensor(data).to(device)  # 将音频数据移动到指定的设备(GPU)

        with torch.no_grad():  # 上下文管理器，确保在推理过程中不会进行梯度计算。
            feature_train = model.extract_features(source=data, output_layer=12)[0]  # 提取音频数据的特征（提取第12层的特征）
        feature_train = torch.squeeze(feature_train, dim=0)
        print("a：",feature_train.shape)


       #将特征和标签组成元组并存储到当前结果列表中
        result = (feature_train,label_train)
        current_results_list_train.append(result)


    ##将当前文件处理后的结果列表存储到字典中
    result_dict['Train'] = current_results_list_train
    # print("result:",result_dict)



    for test_sample in test_indices:
        filename_test, label_test = test_sample
        # print("文件名:", filename_test, "标签:", label_test)
        # print(len(test_indices))
        data_test, _ = librosa.load(filename_test, sr=16000)

        data_t = data_test
        if len(data_t) <= len_wav:  # 检测长度，是否小于等于32000个样本点
            pad_width = len_wav - len(data_t)
            data_t = np.pad(data_t, (0, pad_width), mode='constant', constant_values=0)  # 末尾补零
            # print("a",len(data_m))

        else:
            data_t = data_t[0:len_wav]
            # print(len(data_m))

        data_tw = data_t
        data = data_tw[np.newaxis, :]  # 将音频数据转换为pytorch张量，并添加一个新的维度作为批处理维度。
        data = torch.Tensor(data).to(device)  # 将音频数据移动到指定的设备(GPU)


        with torch.no_grad():  # 上下文管理器，确保在推理过程中不会进行梯度计算。
            feature_test = model.extract_features(source=data, output_layer=12)[0]  # 提取音频数据的特征（提取第12层的特征）
        feature_test = torch.squeeze(feature_test, dim=0)  # 去除特征张量的批处理维度





        # 将特征和标签组成元组并存储到当前结果列表中
        result_test = (feature_test,label_test)
        current_results_list_test.append(result_test)
        # print("feature", feature_test, feature_test.shape)
        # print("label:", result_test)

    ##将当前文件处理后的结果列表存储到字典中
    result_dict['Test'] = current_results_list_test


    ##构建要保存的文件名

    fold_filename = os.path.join(model_path,f"fold_wavlm_label_{num_i}.pkl")
    with open(fold_filename, 'wb') as f:
         pickle.dump(result_dict, f)









