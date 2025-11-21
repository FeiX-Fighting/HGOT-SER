"""
五折交叉验证（每一种情感均分成5份，
Fold_1_1.pkl  第一种情感，分为训练集和测试集（160和40），第一份作为测试集
Fold_1_2.pkl 第一种情感，分为训练集和测试集，第二份作为测试集（160和40）
）

"""

import os
import random
import pickle



emotion_map = {'neutral':0,'happy':1,'sad':2,'angry':3,'fear':4,'surprise':5}
# 定义父文件夹和k值
parent_folder = r'CASIA'

k = 5
##创建文件夹用于保存.pkl文件
fold_path = "fold_output"
os.makedirs(fold_path, exist_ok=True)
# 定义一个空字典用于存储每个折的训练集和测试集
data_dict = {}


# 遍历父文件夹内的所有子文件夹
subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

# print("sub:", subfolders)

##
for i , folder in enumerate(subfolders):
    wav_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
    emotion_label = os.path.basename(folder)
    print("emtion_:",emotion_label)
    random.shuffle(wav_files)
    ##分配标签
    wav_label_tuples =[(wav_file,emotion_map[emotion_label]) for wav_file in wav_files]
    data_dict[emotion_label] = wav_label_tuples    #将.wav文件以字典形式存储
#
print("data:",data_dict)

##定义函数用于生成K折交叉验证的训练和测试集
def generate_folds(random_all):
    folds = []

    for i_f in range(k):  #k次
        train_indices = []
        test_indicex = []
        print("i_f:",i_f)

        for label,indicex in random_all.items():
            ##动态调整折中的数据数量，确保尽可能地利用所有数据
            print("label:",label)
            fold_size = len(indicex) // k + (i_f < len(indicex) % k)
            start = i_f * fold_size
            end = (1+i_f)*fold_size
            print("start:",start)
            print("end:",end)
            test_fold = indicex[start:end]   #测试集
            train_fold = [idx for idx in indicex if idx not in test_fold]   #训练集
            train_indices.extend(train_fold)
            test_indicex.extend(test_fold)
        folds.append((train_indices,test_indicex))

    return folds
#
folds = generate_folds(data_dict)

##定义保存数据的文件夹路径
fold_path = "wav_label_m"
os.makedirs(fold_path, exist_ok=True)
# print("folds:",folds)
# 输出每一折的训练和测试集索引
for fold, (train_indices, test_indices) in enumerate(folds):
    print(f"Fold {fold+1}:")
    print("训练集索引:", len(train_indices))
    print("测试集索引:", len(test_indices))
    fold_to_load = fold+1  #要加载的折数
    fold_filename = os.path.join(fold_path,f"fold_wav_label_{fold_to_load}.pkl")
    with open(fold_filename, 'wb') as f:
        fold_data = {'train':train_indices,'test':test_indices}
        pickle.dump(fold_data, f)

    print(f"Fold{fold+1}的数据已保存到文件：",fold_filename)





#