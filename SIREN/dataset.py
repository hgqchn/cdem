import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import random
import math
import utils
import dem_data_convert

from utils import to_pixel_samples_tensor,torch_imresize


def value_denorm(norm_value, trans):

    # value: B,L,1
    # trans: B,3

    # 确保输入张量在同一设备上
    if norm_value.device != trans.device:
        trans = trans.to(norm_value.device)

    # B 3->B 1-> B 1 1
    data_min= trans[:, 0].reshape(-1, 1, 1)
    norm_min= trans[:, 1].reshape(-1, 1, 1)
    scale= trans[:, 2].reshape(-1, 1, 1)
    # B L 1
    new_data= (norm_value - norm_min) / scale + data_min
    return new_data

class DEMFolder(Dataset):
    def __init__(self, dem_dir, norm_range=(-1, 1), epsilon=1e-6,minmax_height=None,return_name=False,
                 ):
        super(DEMFolder, self).__init__()
        self.dem_list = utils.get_dem_paths_all(dem_dir)
        self.norm_range = norm_range
        self.epsilon = epsilon

        self.return_name = return_name
        # 使用传入的min max值进行归一化，而不是每个dem分别maxmin归一化
        self.minmax_height=minmax_height


    def __getitem__(self, index):
        dem_file=self.dem_list[index]
        filename = utils.get_filename(dem_file)

        data=utils.read_dem(dem_file)

        # 64 64 -> 1 64 64
        data_t = torch.from_numpy(data).unsqueeze(0).float()
        data_norm, trans = dem_data_convert.tensor_maxmin_norm(data_t, self.norm_range, self.epsilon,self.minmax_height)

        if self.return_name:
            return data_norm,trans,filename
        else:
            # data_norm 1 64 64
            # trans 3
            return data_norm,trans

    def __len__(self):
        return len(self.dem_list)

class ImplicitDownsampled(Dataset):
    def __init__(self,
                 dataset, #DEMFolder 类
                 scale=4, # >1
                 ) -> None:
        '''
        Distribute the elevation value at (x,y)

        '''
        super().__init__()
        self.dataset = dataset
        self.length = len(dataset)

        self.scale = scale

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # hr_img 1,H,W
        # trans 3
        # img为归一化后的数据，trans保留了归一化参数
        hr_img,trans = self.dataset[idx]

        assert hr_img.shape[-1]==hr_img.shape[-2]
        assert hr_img.shape[-1]%self.scale==0
        #坐标，值对
        # hr_coord: H*W,2 范围-1,1
        # hr_value: H*W,1
        hr_coord, hr_value =utils.to_pixel_samples_tensor(hr_img.contiguous())

        lr=F.interpolate(hr_img.unsqueeze(0),scale_factor=1/self.scale,mode='nearest').squeeze(0)

        # lr: 1,H//scale,W//scale
        # hr_coord: H*W,2
        # hr_value: H*W,1
        # 3
        return lr,hr_coord,hr_value,trans

class DEMImplicit_Folder_pair(Dataset):
    def __init__(self, hr_dir,lr_dir, norm_range=(-1, 1), epsilon=1e-6,minmax_height=None
                 ):
        super(DEMImplicit_Folder_pair, self).__init__()
        self.dem_hrlist = utils.get_dem_paths_all(hr_dir)
        self.dem_lrlist = utils.get_dem_paths_all(lr_dir)

        self.norm_range = norm_range
        self.epsilon = epsilon


        # 使用传入的min max值进行归一化，而不是每个dem分别maxmin归一化
        self.minmax_height=minmax_height

    def __getitem__(self, index):
        hr_file=self.dem_hrlist[index]
        lr_file=self.dem_lrlist[index]
        filename = utils.get_filename(hr_file)
        lr_filename= utils.get_filename(lr_file)
        assert filename==lr_filename

        hr=utils.read_dem(hr_file)
        lr=utils.read_dem(lr_file)

        # 64 64 -> 1 64 64
        hr_t = torch.from_numpy(hr).unsqueeze(0).float()
        lr_t = torch.from_numpy(lr).unsqueeze(0).float()

        # # 训练用的这个,错的
        # hr_norm, trans = dem_data_convert.tensor_maxmin_norm(hr_t, self.norm_range, self.epsilon,self.minmax_height)
        # lr_norm = dem_data_convert.tensor_maxmin_trans(lr_t, trans)
        # 应该用这个
        lr_norm, trans = dem_data_convert.tensor_maxmin_norm(lr_t, self.norm_range, self.epsilon,self.minmax_height)
        hr_norm = dem_data_convert.tensor_maxmin_trans(hr_t, trans)

        #坐标，值对
        # hr_coord: H*W,2 范围-1,1
        # hr_value: H*W,1
        hr_coord, hr_value =utils.to_pixel_samples_tensor(hr_norm.contiguous())

        return lr_norm,hr_norm,hr_coord,hr_value,trans,filename

    def __len__(self):
        return len(self.dem_hrlist)


if __name__ == '__main__':
    test_dataset = DEMImplicit_Folder_pair(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test',
                                           r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m'
                            )

    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             num_workers=4)

    data=next(iter(test_loader))

    hr=data[1]
    hr_value=data[3]
    trans= data[4]
    hr_=dem_data_convert.tensor4D_maxmin_denorm(hr,trans)
    hr_denorm = value_denorm(hr_value, trans)
    hr1_denorm = hr_denorm[0]
    hr1_denorm = hr1_denorm.view(1,64,64)
    hr1=hr_[0]
    torch.equal(hr1, hr1_denorm)
    pass
