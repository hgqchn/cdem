import sys
import os
import torch
import torch.nn as nn
import numpy as np
import utils

# 其实就是边缘检测罢了，sobel算子
class Slope_map(nn.Module):
    def __init__(self):

        super(Slope_map, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        # -1 0 1
        # -2 0 2
        # -1 0 -1
        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1
        # -1 -2 -1
        # 0 0 0
        # 1 2 1
        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))

        # nn.Parameter 注册为模型参数
        self.weight1 =nn.Parameter(torch.tensor(weight1)) # 自定义的权值
        self.weight2 =nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        self.bias.requires_grad = False
    def forward(self, x):
        # x 为归一化的输入
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy= torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        return dx,dy


if __name__ == '__main__':

    weight1 = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    weight1 = np.reshape(weight1, (1, 1, 3, 3))
    weight1 = nn.Parameter(torch.tensor(weight1))
    bias = nn.Parameter(torch.zeros(1))

    test_input=torch.tensor([[[1, 2, 3],
                            [4, 5, 6],
                             [7, 8, 9]]], dtype=torch.float32)
    test_input=test_input.view(1, 1, 3, 3)

    #res=torch.conv2d(test_input, weight1,bias, stride=1, padding=1)

    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    hr_dem = utils.read_dem(hr_file_path)
    hr_dem = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)

    slope_map= Slope_map()
    hr_dx,hr_dy=slope_map(hr_dem)

    res = torch.conv2d(hr_dem, weight1, bias, stride=1, padding=1)
    print(res[0,0,:4,:4])

    hr_np=hr_dem.squeeze().squeeze()
    pass
