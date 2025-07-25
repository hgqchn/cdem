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
    hr_coord = utils.get_pixel_center_coord_tensor((hr_height, hr_width))
    hr_coord = hr_coord.to(device).unsqueeze(0)  # Add batch dimension

    out=model(hr_coord)
    coord_grad=out['model_in']
    model_out=out['model_out']


    lr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m\dem_0\dem0_0.TIF'
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    hr_dem = utils.read_dem(hr_file_path)
    hr_dem = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)

    lr_dem = utils.read_dem(lr_file_path)
    lr_dem = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    bic_dem = F.interpolate(lr_dem, scale_factor=scale, mode='bicubic')

    slope_map = Slope_map()
    hr_slope,hr_dx,hr_dy=slope_map(hr_dem)
    hr_slope=hr_slope[...,1:-1,1:-1]  # 去掉边缘
    hr_dx=hr_dx[...,1:-1,1:-1]
    hr_dy=hr_dy[...,1:-1,1:-1]

    bic_slope,bic_dx,bic_dy=slope_map(bic_dem)
    bic_slope=bic_slope[...,1:-1,1:-1]
    bic_dx=bic_dx[...,1:-1,1:-1]
    bic_dy=bic_dy[...,1:-1,1:-1]

    bic_slope_mae=torch.mean(torch.abs(hr_slope-bic_slope))
    bic_dx_mae= torch.mean(torch.abs(hr_dx-bic_dx))
    bic_dy_mae= torch.mean(torch.abs(hr_dy-bic_dy))

    # 模型运算
    _, trans = dem_data_convert.tensor_maxmin_norm(lr_dem, (-1, 1), 1e-6,
                                                       None)
    sr_value = dataset.value_denorm(model_out, trans)

    dydx_0=gradient(model_out, coord_grad, grad_outputs=torch.ones_like(model_out))
    # 1,H*W,2
    # dx,dy
    dydx= gradient(sr_value, coord_grad, grad_outputs=torch.ones_like(model_out))
    dydx_denorm=2/64*dydx

    sr_dem = sr_value.view(1, 1, hr_height, hr_width).cpu()
    sr_dem_slope,sr_dem_dx,sr_dem_dy=slope_map(sr_dem)
    sr_dem_slope=sr_dem_slope[...,1:-1,1:-1]
    sr_dem_dx=sr_dem_dx[...,1:-1,1:-1]
    sr_dem_dy=sr_dem_dy[...,1:-1,1:-1]

    dx_,dy_=dydx_denorm[...,1],dydx_denorm[...,0]
    dx_=dx_.view(1, 1, hr_height, hr_width).cpu()
    dy_=dy_.view(1, 1, hr_height, hr_width).cpu()
    dx_=dx_[...,1:-1,1:-1]
    dy_=dy_[...,1:-1,1:-1]

    hr_dem=hr_dem[...,1:-1,1:-1]

    # 计算插值的误差
    bic_dem= bic_dem[...,1:-1,1:-1]
    bic_dem_mae= torch.mean(torch.abs(hr_dem - bic_dem))
    print(f"bic_dem_mae: {bic_dem_mae.item():.4f}\n"
          f"bic_slope_mae: {bic_slope_mae.item():.4f}\n"
          f"bic_dx_mae: {bic_dx_mae.item():.4f}\n"
          f"bic_dy_mae: {bic_dy_mae.item():.4f}\n")

    # 计算模型超分得到的SR的误差
    sr_dem=sr_dem[...,1:-1,1:-1]
    sr_dem_mae=torch.mean(torch.abs(hr_dem - sr_dem))
    sr_dem_slope_mae=torch.mean(torch.abs(hr_slope - sr_dem_slope))
    sr_dem_dx_mae=torch.mean(torch.abs(hr_dx - sr_dem_dx))
    sr_dem_dy_mae=torch.mean(torch.abs(hr_dy - sr_dem_dy))
    print(f"sr_dem_mae: {sr_dem_mae.item():.4f}\n"
          f"sr_dem_slope_mae: {sr_dem_slope_mae.item():.4f}\n"
          f"sr_dem_dx_mae: {sr_dem_dx_mae.item():.4f}\n"
          f"sr_dem_dy_mae: {sr_dem_dy_mae.item():.4f}\n")

    # 模型求导结果
    dx_mae=torch.mean(torch.abs(hr_dx - dx_))
    dy_mae=torch.mean(torch.abs(hr_dy - dy_))
    print(f"dx_mae: {dx_mae.item():.4f}\n"
          f"dy_mae: {dy_mae.item():.4f}\n")

    # eval_res = dem_features.cal_DEM_metric(sr_dem, hr_dem, device=device)


#     import numpy as np
#     save_path=r'D:\codes\cdem\output\SIREN\test\test_dem'
#     save_file_path=os.path.join(save_path,"trans.csv")
#     trans_np=trans.cpu().numpy()
#     np.savetxt(save_file_path,trans_np,delimiter=',',header='min_value, norm_min, scale',comments='')
# #   bic_dem=bic_dem.squeeze(0).squeeze(0).numpy()
#
#     import imageio
#     imageio.imwrite(os.path.join(save_path, "bic_dem.tif"), bic_dem, format='TIFF')





    pass
