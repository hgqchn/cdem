import sys
import os

import utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import dem_features

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 单个DEM
# HR,LR,Bic
if __name__ == '__main__':

    # test_dem=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_2\dem2_1026.TIF'
    # test_dem=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_2\dem2_969.TIF'
    # test_dem=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_1\dem1_1110.TIF'
    test_dem = r"D:\Data\DEM_data\测试用\4096and8192\8_106_18_L14.tif"
    dem=utils.read_dem(test_dem)

    hr_dem=dem[:512,:512] # 截取前128x128的区域
    lr_dem=utils.cv2_imresize(hr_dem,1/8,mode="nearest")
    bic_dem=utils.torch_imresize(lr_dem,4,mode="bicubic")

    # plt.figure()
    # plt.imshow(lr_dem,cmap='terrain')
    # plt.axis('off')
    # plt.savefig('lr_dem.png', bbox_inches='tight', pad_inches=0)
    # plt.close()
    # plt.figure()
    # plt.imshow(hr_dem,cmap='terrain')
    # plt.axis('off')
    # plt.savefig('hr_dem.png', bbox_inches='tight', pad_inches=0)
    # plt.close()

    # 差分方法计算
    slope_map = dem_features.Slope_torch(pixel_size=30)

    # hr
    hr_4dtensor = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)
    hr_dx,hr_dy,hr_slope=slope_map(hr_4dtensor,return_dxdy=True)
    hr_slope=hr_slope[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()  # 去掉边缘
    hr_dx=hr_dx[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()
    hr_dy=hr_dy[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()

    plt.figure()
    plt.imshow(hr_dx)
    plt.axis('off')
    plt.savefig('hd_dx.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.figure()
    plt.imshow(hr_dy)
    plt.axis('off')
    plt.savefig('hr_dy.png', bbox_inches='tight', pad_inches=0)
    plt.close()

