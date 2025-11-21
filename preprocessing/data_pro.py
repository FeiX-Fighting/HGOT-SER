#!/usr/bin/env python3
"""
单人一键运行版——路径全部写死，用法同以前：
python extract.py
"""
import os
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import librosa
from WavLM.WavLM import WavLM, WavLMConfig

# =========  只改这一坨就行  ==========
OUT_DIR = r"output"          # 当前文件夹下的 output 子文件夹
Path(OUT_DIR).mkdir(exist_ok=True)
WAVLM_CKPT = r"WavLM-Large.pt"
LABEL_JSON = r"label_1.json"
WAV_DIR = r"Audio_16k"
#OUT_DIR = r"."               # 想换输出目录就改这里
LEN_WAV = 96000             # 采样点长度
# =====================================


def load_wavlm(ckpt_path,audio, device):
    checkpoint = torch.load(ckpt_path)
    cfg = WavLMConfig(checkpoint["cfg"])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    ##wavlm特征
    with torch.no_grad():  # 上下文管理器，确保在推理过程中不会进行梯度计算。
        feature = model.extract_features(source=audio, output_layer=12)[0]  # 提取音频数据的特征（提取第12层的特征）
    feature = torch.squeeze(feature, dim=0)  # 去除特征张量的批处理维度
    return feature




@torch.no_grad()
def extract_feature(audio,  fixed_len):
    if len(audio) <= fixed_len:
        audio_feat = np.pad(audio, (0, fixed_len - len(audio)))
    else:
        audio_feat = audio[:fixed_len]

    return audio_feat


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(LABEL_JSON, "r", encoding="utf-8") as f:
        labels = json.load(f)

    wav_dir = Path(WAV_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(exist_ok=True)

    sessions = {f"Session{i}": [] for i in range(1, 6)}

    for split in ("Train", "Test"):
        for key, emotion in labels[split].items():
            type_n = split

            sess = key[:5]
            print("s",sess)
            wav_path = wav_dir / key

            if not wav_path.exists():
                print(f"[WARN] skip missing file: {wav_path}")
                continue

            audio, _ = librosa.load(wav_path, sr=16000)
            audio_feat = extract_feature(audio, LEN_WAV)
            data_feat_GPU = audio_feat[np.newaxis, :]
            data_feat_GPU = torch.Tensor(data_feat_GPU).to(device)
            wavlm_feat = load_wavlm(WAVLM_CKPT,data_feat_GPU, device)

            result = (wavlm_feat,  key, labels[type_n][key])
            session_key = f"Session{int(sess[-2:])}"

            sessions[session_key].append(result)

    out_file = out_dir / f"Session_large_dir_{LEN_WAV}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(sessions, f)
    print(f"[OK] 已保存 -> {out_file.resolve()}")


if __name__ == "__main__":
    main()