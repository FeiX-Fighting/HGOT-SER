import os
import joblib
from collections import defaultdict

# ========== 1. 配置 ==========
root_dir = r'\CASIA数据库\CASIA_speaker'   # 数据根目录
emotion_map = {'neutral': 0, 'happy': 1, 'sad': 2,
               'angry': 3, 'fear': 4, 'surprise': 5}
out_dir = 'CASIA_folds'                            # 保存 fold 的文件夹
os.makedirs(out_dir, exist_ok=True)                # 若不存在则创建

# ========== 2. 扫描所有 wav ==========
all_wavs = {}
for sub in sorted(os.listdir(root_dir)):
    sub_path = os.path.join(root_dir, sub)
    if not os.path.isdir(sub_path):
        continue
    all_wavs[sub] = [os.path.join(sub_path, f)
                     for f in os.listdir(sub_path)
                     if f.lower().endswith('.wav')]
subs = sorted(all_wavs.keys())

# ========== 3. 标签提取 ==========
def extract_label(fp):
    """文件名格式：xxx-emotion-xxx.wav，取第二段"""
    name = os.path.basename(fp)[:-4]      # 去掉 .wav
    parts = name.split('-')
    if len(parts) < 2:
        return None
    return emotion_map.get(parts[1].lower())   # 不在 map 返回 None

# ========== 4. 4 折留一划分 ==========
folds = []
for test_sub in subs:
    train, test = [], []
    # ---- 测试集 ----
    for fp in all_wavs[test_sub]:
        lbl = extract_label(fp)
        if lbl is None:
            print(f'[WARN] 跳过测试文件（未知标签）: {fp}')
            continue
        test.append((fp, lbl))
    # ---- 训练集 ----
    for train_sub in subs:
        if train_sub == test_sub:
            continue
        for fp in all_wavs[train_sub]:
            lbl = extract_label(fp)
            if lbl is None:
                print(f'[WARN] 跳过训练文件（未知标签）: {fp}')
                continue
            train.append((fp, lbl))
    folds.append({'train': train, 'test': test})

# ========== 5. 保存与输出数量 ==========
for idx, fold in enumerate(folds):
    train_n, test_n = len(fold['train']), len(fold['test'])
    print(f'fold_{idx}:  train={train_n}, test={test_n}')
    joblib.dump(fold, os.path.join(out_dir, f'wav_label_{idx}.pkl'))

    print('  train 前 5 条:')
    for item in fold['train'][:5]:
        print(f'    {item}')
    print('  test  前 5 条:')
    for item in fold['test'][:5]:
        print(f'    {item}')
    print()

print(f'全部完成！4 个 pkl 已保存到 ./{out_dir}/')