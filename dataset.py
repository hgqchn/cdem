import os
import sys
import h5py
import cv2
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

import utils
from utils import read_dem,torch_imresize,get_dem_paths_all
import dem_data_convert
from functools import partial

nearest_torch_method = partial(torch_imresize, mode="nearest")
bicubic_torch_method = partial(torch_imresize, mode="bicubic")

def crop_img_numpy(img,scale,mode='default'):
    """

    :param img: H,W,C or H,W numpy array
    :param scale: > 1
    :param mode:
    :return:
    """
    width=img.shape[1]
    height=img.shape[0]
    lr_width=img.shape[1]//scale
    lr_height=img.shape[0]//scale
    new_width=lr_width*scale
    new_height=lr_height*scale
    h_start = 0
    w_start = 0
    h_end=height
    w_end=width
    if mode=='default':
        h_end=new_height
        w_end=new_width
    elif mode=="center":
        h_start=(height-new_height)//2
        w_start=(width-new_width)//2
        h_end=h_start+new_height
        w_end=w_start+new_width

    new_img=img[h_start:h_end,w_start:w_end]
    return new_img


class DEMDatasetFolder_from_gt(Dataset):
    """
    only exist gt(hr) dem
    get dem tensor or dem file path

    """

    def __init__(self, hr_dir, scale=2, mode="bilinear", norm_range=(-1, 1), epsilon=10,minmax_height=None):
        super(DEMDatasetFolder_from_gt, self).__init__()
        self.hr_img_list = get_dem_paths_all(hr_dir)
        self.scale = scale
        self.mode = mode
        self.norm_range = norm_range
        self.epsilon = epsilon
        # 使用传入的min max值进行归一化，而不是每个dem分别maxmin归一化
        self.minmax_height=minmax_height


    def __getitem__(self, index):
        label = read_dem(self.hr_img_list[index])
        label = crop_img_numpy(label, self.scale)
        filename = utils.get_filename(self.hr_img_list[index])

        # 64 64 -> 1 1 64 64
        label_t = torch.from_numpy(label).unsqueeze(0).unsqueeze(0).float()

        data_t = F.interpolate(label_t, scale_factor=1. / self.scale, mode=self.mode)
        # 1 64 64
        label_t = label_t.squeeze(0)
        data_t = data_t.squeeze(0)


        data_norm, trans = dem_data_convert.tensor_maxmin_norm(data_t, self.norm_range, self.epsilon,self.minmax_height)
        label_norm=dem_data_convert.tensor_maxmin_trans(label_t,trans)

        tensor_data = data_norm, label_norm, trans
        return tensor_data, filename

    def __len__(self):
        return len(self.hr_img_list)


tfasr_120to30_train_down_nearest = DEMDatasetFolder_from_gt(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Train', scale=4,
                                               mode="nearest", norm_range=(-1, 1), epsilon=10)
tfasr_120to30_test_down_nearest = DEMDatasetFolder_from_gt(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test', scale=4,
                                              mode="nearest", norm_range=(-1, 1), epsilon=10)

tfasr30_minmax=(0,3800)
tfasr_120to30_train_down_nearest_v2 = DEMDatasetFolder_from_gt(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Train', scale=4,
                                               mode="nearest", norm_range=(-1, 1), epsilon=10,minmax_height=tfasr30_minmax)
tfasr_120to30_test_down_nearest_v2 = DEMDatasetFolder_from_gt(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test', scale=4,
                                              mode="nearest", norm_range=(-1, 1), epsilon=10,minmax_height=tfasr30_minmax)


if __name__ == '__main__':
    pass
