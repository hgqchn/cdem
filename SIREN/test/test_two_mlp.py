import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from SIREN import meta_modules,dataset,modules
from utils import to_pixel_samples_tensor,torch_imresize
import utils
import dem_data_convert,dem_features

import matplotlib.pyplot as plt

# 提前初始化 CUDA 上下文
torch.cuda.init()

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# 其实就是边缘检测罢了，sobel算子
class Slope_map(nn.Module):
    def __init__(self):

        super(Slope_map, self).__init__()


        weight1=np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=np.float32)
        weight2=np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=np.float32)

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))

        # weight1=weight1/8
        # weight2=weight2/8

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
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))

        return slope,dx,dy



if __name__ == '__main__':

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    weight_path1=r"D:\codes\cdem\output\SIREN\test\2025-05-29_16-23-22\mlp_params\dem0_0_params.pth"
    model1=modules.SimpleMLPNet(hidden_features=64,num_hidden_layers=3,image_resolution=(16,16))
    sd1=torch.load(weight_path1,weights_only=True)
    model1.load_state_dict(sd1)
    model1.to(device)
    model1.eval()

    weight_path2=r"D:\codes\cdem\test_fit_data\out\siren_dem0_0.pth"
    model2=modules.SimpleMLPNet(hidden_features=64,num_hidden_layers=3,image_resolution=(16,16))
    sd2=torch.load(weight_path2,weights_only=True)
    model2.load_state_dict(sd2)
    model2.to(device)
    model2.eval()

    scale=4
    imagesize=64
    lr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m\dem_0\dem0_0.TIF'
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    hr_dem = utils.read_dem(hr_file_path)
    hr_dem = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)
    lr_dem = utils.read_dem(lr_file_path)
    lr_dem = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    bic_dem = F.interpolate(lr_dem, scale_factor=scale, mode='bicubic')

    coord_idx=torch.stack(torch.meshgrid(torch.arange(1,imagesize+1), torch.arange(1,imagesize+1),indexing='ij'), dim=-1)
    coord_idx=coord_idx.view(-1,2).float()
    coord_idx=coord_idx.to(device).unsqueeze(0)

    coord_norm = (coord_idx * 2 - 1) / imagesize - 1

    input_coord1,out1,grad1=model1.forward_coord(coord_idx,imagesize=64)
    input_coord2,out2,grad2=model2.forward_coord(coord_idx,imagesize=64)

    model_out = model2(coord_norm)
    out2_1 = model_out['model_out']
    coords = model_out['model_in']
    grad_outputs = torch.ones_like(out2_1)
    grad2_1 = torch.autograd.grad(out2_1, coords, grad_outputs=grad_outputs, create_graph=True)[0]
    grad2_1=grad2_1.detach().cpu()

    grad1=grad1.detach().cpu()
    grad2=grad2.detach().cpu()
    # 模型运算
    _, trans = dem_data_convert.tensor_maxmin_norm(lr_dem, (-1, 1), 1e-6,
                                                       None)
    sr1 = dataset.value_denorm(out1, trans)
    sr1=sr1.view(1,1,imagesize,imagesize).detach().cpu()
    sr2= dataset.value_denorm(out2,trans)
    sr2=sr2.view(1,1,imagesize,imagesize).detach().cpu()

    mae1= torch.abs(sr1 - hr_dem).mean()
    mae2 = torch.abs(sr2 - hr_dem).mean()
    bic_mae=torch.abs(hr_dem-bic_dem).mean()

    # 栅格计算的坡度
    slope_map = Slope_map()
    hr_slope, hr_dx, hr_dy = slope_map(hr_dem)
    hr_slope = hr_slope[..., 1:-1, 1:-1]  # 去掉边缘
    hr_dx = hr_dx[..., 1:-1, 1:-1]
    hr_dy = hr_dy[..., 1:-1, 1:-1]
    bic_slope, bic_dx, bic_dy = slope_map(bic_dem)
    bic_slope = bic_slope[..., 1:-1, 1:-1]
    bic_dx = bic_dx[..., 1:-1, 1:-1]
    bic_dy = bic_dy[..., 1:-1, 1:-1]

    # 求导计算的坡度
    # grad1=grad1/trans[...,2]
    # grad2=grad2/trans[...,2]
    dx1=grad1[...,1]
    dy1=grad1[...,0]
    dx1=dx1.view(1,1,imagesize,imagesize)
    dx1=dx1[...,1:-1,1:-1]
    dy1=dy1.view(1,1,imagesize,imagesize)
    dy1=dy1[...,1:-1,1:-1]


    dx2=grad2[...,1]
    dy2=grad2[...,0]
    dx2=dx2.view(1,1,imagesize,imagesize)
    dx2=dx2[...,1:-1,1:-1]
    dy2=dy2.view(1,1,imagesize,imagesize)
    dy2=dy2[...,1:-1,1:-1]

    dx2_1= grad2_1[..., 1]
    dx2_1= dx2_1.view(1,1,imagesize,imagesize)
    dx2_1=dx2_1[...,1:-1,1:-1]

    fig,axes= plt.subplots(2,4,figsize=(12,10))
    for ax in axes.flatten():
        ax.axis('off')
    axes[0,0].imshow(hr_dem.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='terrain')
    axes[0,0].set_title('HR DEM')
    axes[0,1].imshow(sr1.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='terrain')
    axes[0,1].set_title('SR DEM (MLP)')
    axes[0,2].imshow(sr2.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='terrain')
    axes[0,2].set_title('SR DEM (SIREN)')

    axes[1,0].imshow(hr_dx.squeeze(0).squeeze(0).numpy())
    axes[1,0].set_title('HR DEM dx')
    axes[1,1].imshow(dx1.squeeze(0).squeeze(0).numpy())
    axes[1,1].set_title('MLP DEM dx')
    axes[1,2].imshow(dx2.squeeze(0).squeeze(0).numpy())
    axes[1,2].set_title('SIREN DEM dx')
    axes[1,3].imshow(dx2_1.squeeze(0).squeeze(0).numpy())
    axes[1,3].set_title('SIREN DEM dx_1')

    plt.show(block=True)

    #utils.write_dem(sr2.squeeze(0).squeeze(0).cpu().detach().numpy(),r'D:\codes\cdem\output\SIREN\test\test_dem\sr2.tif')


    pass
