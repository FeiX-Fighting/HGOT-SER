import os
import random
import pickle

emotion_map = {'N':0,'F':1,'T':2,'W':3,'A':4,'E':5,'L':6}
parent_folder = r'classified_wav_files'
k = 5
random_seed = 2024
random.seed(random_seed)

# 1. 读入并按情感保存
data_dict = {}
subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
for folder in subfolders:
    emotion_label = os.path.basename(folder)
    wav_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
    wav_label_tuples = [(wav_file, emotion_map[emotion_label]) for wav_file in wav_files]
    random.shuffle(wav_label_tuples)  # 按情感打乱
    data_dict[emotion_label] = wav_label_tuples

def split_into_k_folds(lst, k):
    """把列表平均分成 k 份，前面几份多一个"""
    n = len(lst)
    folds = []
    base = n // k
    rem = n % k
    start = 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        end = start + size
        folds.append(lst[start:end])
        start = end
    return folds

# 2. 为每个情感生成 k 份，再按折号组装 train/test
label_folds = {}  # 每个情感 -> [fold0, fold1, ..., fold4]
for label, items in data_dict.items():
    label_folds[label] = split_into_k_folds(items, k)

folds = []
for i_f in range(k):
    train_set, test_set = [], []
    for label, folds_for_label in label_folds.items():
        test_part = folds_for_label[i_f]
        train_parts = [x for j, x in enumerate(folds_for_label) if j != i_f]
        train_flat = [item for part in train_parts for item in part]
        test_set.extend(test_part)
        train_set.extend(train_flat)
    folds.append((train_set, test_set))

# 3. 保存
fold_path = "wav_label_m"
os.makedirs(fold_path, exist_ok=True)

for fold_idx, (train_indices, test_indices) in enumerate(folds, start=1):
    fold_filename = os.path.join(fold_path, f"fold_wav_label_{fold_idx}.pkl")
    with open(fold_filename, 'wb') as f:
        fold_data = {'train': train_indices, 'test': test_indices}
        pickle.dump(fold_data, f)
    print(f"Fold {fold_idx} saved to {fold_filename}, "
          f"train={len(train_indices)}, test={len(test_indices)}")
