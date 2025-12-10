import pickle

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
# 
#
# with open(path, 'rb') as f:
#     data = pickle.load(f)
IEMOCAP_dir = {'neutral': 0, 'happy': 1, 'sad': 2, 'anger': 3}
data_list = ['Session1','Session2','Session3','Session4','Session5']

class mode_dataset(Dataset):
    def __init__(self,out_path,mode_number,mode):
        #读取文件
        print(out_path+'/'+'Session_large_dir.pkl')
        with open(out_path+'/'+'Session_large_dir.pkl', 'rb') as f1:
            data = pickle.load(f1)   #特征向量和标签相结合
        #
        print("aaa")
        if mode_number == 1:
            if mode == 'test':
                self.data = data[data_list[0]]
            if mode == 'train':
                self.data = data[data_list[1]]+data[data_list[2]]+data[data_list[3]]+data[data_list[4]]
        if mode_number == 2:
            if mode == 'test':
                self.data = data[data_list[1]]
            if mode == 'train':
                self.data = data[data_list[0]]+data[data_list[2]]+data[data_list[3]]+data[data_list[4]]
        if mode_number == 3:
            if mode == 'test':
                self.data = data[data_list[2]]
            if mode == 'train':
                self.data = data[data_list[1]]+data[data_list[0]]+data[data_list[3]]+data[data_list[4]]
        if mode_number == 4:
            if mode == 'test':
                self.data = data[data_list[3]]
            if mode == 'train':
                self.data = data[data_list[1]]+data[data_list[2]]+data[data_list[0]]+data[data_list[4]]
        if mode_number == 5:
            if mode == 'test':
                self.data = data[data_list[4]]
            if mode == 'train':
                self.data = data[data_list[1]]+data[data_list[2]]+data[data_list[3]]+data[data_list[0]]

        self.len = len(self.data)

    def __getitem__(self,index):   #定义类的方法，按索引获取数据集中的样本
        wavlm_data_inx = self.data[index][0]   #
        mfcc_data_inx = self.data[index][1]  #
        # mfcc_data_inx = self.data[index][1]  #
        label_inx = IEMOCAP_dir[self.data[index][3]]  #
        
        # print(wavlm_data_inx.shape)
        # print(type(wavlm_data_inx))
        # print(mfcc_data_inx.shape)
        # print(label_inx)
        # print(aaaaaaa)
        
        mfcc_data_inx = np.transpose(mfcc_data_inx, (1, 0)) 

    
        mfcc_data_inx = torch.from_numpy(mfcc_data_inx)      #转换为张量
        
        label_inx = np.array(label_inx)
        label_inx = torch.from_numpy(label_inx)

        return wavlm_data_inx,mfcc_data_inx,label_inx   #返回数据特征集和数据标签
    def __len__(self):
        return self.len


