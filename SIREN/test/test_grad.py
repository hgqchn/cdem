import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt

from SIREN import meta_modules,dataset,modules
from utils import to_pixel_samples_tensor,torch_imresize
import utils
import dem_data_convert,dem_features

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
        weight1=weight1/8
        weight2=weight2/8

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



    weight_path=r"D:\codes\cdem\output\SIREN\test\2025-05-29_16-23-22\mlp_params\dem0_0_params.pth"
    model=modules.SimpleMLPNet(hidden_features=64,num_hidden_layers=3,image_resolution=(16,16))
    sd=torch.load(weight_path,weights_only=True)
    # # sd 是批次的
    # for name,params in sd.items():
    #     sd[name].squeeze_(0)
    # torch.save(sd,weight_path)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    hr_height=64
    hr_width=64
    scale=4

    lr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m\dem_0\dem0_0.TIF'
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    hr_dem = utils.read_dem(hr_file_path)
    hr_dem = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)
    lr_dem = utils.read_dem(lr_file_path)
    lr_dem = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    bic_dem = F.interpolate(lr_dem, scale_factor=scale, mode='bicubic')

    hr_coord = utils.get_pixel_center_coord_tensor((hr_height, hr_width))
    hr_coord = hr_coord.to(device).unsqueeze(0)  # Add batch dimension

    out=model(hr_coord)
    coord_grad=out['model_in']
    model_out=out['model_out']

    coord_idx=torch.stack(torch.meshgrid(torch.arange(1,hr_height+1), torch.arange(1,hr_width+1),indexing='ij'), dim=-1)
    coord_idx=coord_idx.view(-1,2).float()
    coord_idx=coord_idx.to(device).unsqueeze(0)

    input_coord,new_out,new_grad=model.forward_coord(coord_idx,imagesize=64)


    # 模型运算
    _, trans = dem_data_convert.tensor_maxmin_norm(lr_dem, (-1, 1), 1e-6,
                                                       None)
    sr_value = dataset.value_denorm(model_out, trans)

    new_sr_value=dataset.value_denorm(new_out, trans)
    new_dydx=gradient(new_sr_value, input_coord, grad_outputs=torch.ones_like(new_sr_value))
    new_sr= new_sr_value.view(1, 1, hr_height, hr_width).cpu()
    new_dx_,new_dy_=new_dydx[...,1],new_dydx[...,0]
    new_dx_=new_dx_.view(1, 1, hr_height, hr_width).cpu()
    new_dy_=new_dy_.view(1, 1, hr_height, hr_width).cpu()

    sr_dem = sr_value.view(1, 1, hr_height, hr_width).cpu()
    slope_map = Slope_map()
    sr_dem_slope,sr_dem_dx,sr_dem_dy=slope_map(sr_dem)


    fig,axes=plt.subplots(1,3)
    for ax in axes:
        ax.axis('off')
    axes[0].imshow(hr_dem.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='terrain')
    axes[0].set_title('HR DEM')
    axes[1].imshow(sr_dem.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='terrain')
    axes[1].set_title('SR DEM')
    axes[2].imshow(bic_dem.squeeze(0).squeeze(0).numpy(), cmap='terrain')
    axes[2].set_title('Bicubic DEM')

    fig,axes=plt.subplots(1,2)
    for ax in axes:
        ax.axis('off')
    ax1=axes[0].imshow((hr_dem-sr_dem).squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray')
    axes[0].set_title('SR DEM error')

    axes[1].imshow((hr_dem-bic_dem).squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='gray')
    axes[1].set_title('Bic DEM error')

    fig.colorbar(ax1, ax=axes[0])



    sr_error=(hr_dem-sr_dem).squeeze(0).squeeze(0).cpu().detach().numpy()
    bic_error=(hr_dem-bic_dem).squeeze(0).squeeze(0).cpu().detach().numpy()
    sr_error_sum=np.sum(np.abs(sr_error))
    bic_error_sum=np.sum(np.abs(bic_error))

    sr_dem_dx_np=sr_dem_dx.squeeze(0).squeeze(0).cpu().detach().numpy()
    sr_dem_dx_show=sr_dem_dx_np[1:-1, 1:-1]  # 去除边缘
    new_dx_np=new_dx_.squeeze(0).squeeze(0).cpu().detach().numpy()

    fig,axes=plt.subplots(1,3,figsize=(13, 3))
    for ax in axes:
        ax.axis('off')
    pos1=axes[0].imshow(hr_dem.squeeze(0).squeeze(0).cpu().detach().numpy(), cmap='terrain')
    axes[0].set_title('HR DEM')
    fig.colorbar(pos1, ax=axes[0])
    pos2=axes[1].imshow(sr_dem_dx_show, cmap='viridis')
    axes[1].set_title('SR DEM dx')
    fig.colorbar(pos2, ax=axes[1])
    pos3=axes[2].imshow(new_dx_np, cmap='viridis')
    axes[2].set_title('dx')
    fig.colorbar(pos3, ax=axes[2])

    # eval_res = dem_features.cal_DEM_metric(sr_dem, hr_dem, device=device)

    plt.figure()
    plt.imshow(new_dx_np)

#     import numpy as np
#     save_path=r'D:\codes\cdem\output\SIREN\test\test_dem'
#     save_file_path=os.path.join(save_path,"trans.csv")
#     trans_np=trans.cpu().numpy()
#     np.savetxt(save_file_path,trans_np,delimiter=',',header='min_value, norm_min, scale',comments='')
# #   bic_dem=bic_dem.squeeze(0).squeeze(0).numpy()
#
#     import imageio
#     imageio.imwrite(os.path.join(save_path, "bic_dem.tif"), bic_dem, format='TIFF')

    plt.show(block=True)


    pass
