import sys
import os
from collections import OrderedDict

import re
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

import copy

import dem_data_convert
import record
import utils
import dem_features

import matplotlib.pyplot as plt

from SIREN import meta_modules, modules

from SIREN.dataset import DEMFolder, ImplicitDownsampled,DEMImplicit_Folder_pair, value_denorm
from mymodel.model import ImplicitModel
from mymodel.SwinIR import SwinEncoder





if __name__ == '__main__':
    torch.cuda.init()  # 提前初始化 CUDA 上下文
    #torch.cuda.current_device()
    device = utils.default_device

    current_time = utils.get_current_time()
    output_dir=fr'D:\codes\cdem\generate_results\generate_dem\{current_time}'
    utils.make_dir(output_dir)

    lrsize = 16
    scale = 4
    hrsize = lrsize * scale
    hr_res=30

    exp_dir=r'D:\codes\cdem\output\mymodel_swin\2025-07-08_20-55-24'
    ckp_name='best_mymodel_swin_161_2.4091.pth'
    config_file=os.path.join(exp_dir,'config.yaml')
    ckp_file=os.path.join(exp_dir,'checkpoint',ckp_name)
    config=record.get_config_from_yaml(config_file)
    model_config = config['model_config']

    # 模型
    target_config = model_config['target_config']
    target_net = modules.SimpleMLPNetv1(**target_config)

    # -----------------------------------------#
    # dem file

    #dem_name=r'dem6_799.TIF'
    dem_name=r'dem0_0.TIF'
    dem_basename=utils.get_filename(dem_name,with_ext=False)
    dem_idx=re.search(r'dem(\d+)_',dem_name).group(1)

    hr_file= fr'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_{dem_idx}\{dem_name}'
    lr_file= fr'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m\dem_{dem_idx}\{dem_name}'
    mlp_file=fr'D:\codes\cdem\generate_results\结果和参数\mlp_params\{dem_basename}_trans.pth'

    hr_dem= utils.read_dem(hr_file)
    hr_4dtensor = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)
    lr_dem= utils.read_dem(lr_file)
    lr_4dtensor = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    bic_4dtensor = F.interpolate(lr_4dtensor, scale_factor=scale, mode='bicubic')
    bic_dem= bic_4dtensor.squeeze(0).squeeze(0).numpy()
    utils.write_dem(lr_dem, os.path.join(output_dir, f'{dem_basename}_lr.TIF'))
    utils.write_dem(hr_dem, os.path.join(output_dir, f'{dem_basename}_hr.TIF'))
    utils.write_dem(bic_dem, os.path.join(output_dir, f'{dem_basename}_bic.TIF'))

    hr_coord = utils.get_pixel_center_coord_tensor((hrsize, hrsize))
    hr_coord = hr_coord.to(device).unsqueeze(0)  # Add batch dimension

    # -----------------------------------------#
    # sr dem and dy_dx by mlp
    mlp_params=torch.load(mlp_file,weights_only=False)
    sd=mlp_params['state_dict']
    #在cuda上
    trans=mlp_params['trans']
    trans=trans.unsqueeze(0).to(device)
    target_net.load_state_dict(sd)
    target_net.to(device)
    target_net.eval()
    hr_coord.to(device)
    out=target_net(hr_coord,return_grad=True)
    coord_input= out['model_in']
    sr_value = out['model_out']
    dy_dx=out['dy_dx']
    sr_value = value_denorm(sr_value, trans)
    trans_scale=trans[:,2]
    dy_dx=dy_dx/trans_scale
    sr_4dtensor = sr_value.view(1, 1, hrsize, hrsize).detach().cpu()
    sr_dem_np=sr_4dtensor.squeeze().squeeze().numpy()
    utils.write_dem(sr_dem_np,os.path.join(output_dir, f'{dem_basename}_sr.TIF'))
    # -----------------------------------------#
    # 可视化地形图
    all_max = max(np.max(hr_dem), np.max(lr_dem), np.max(bic_dem),np.max(sr_dem_np))
    all_min = min(np.min(hr_dem), np.min(lr_dem), np.min(bic_dem),np.min(sr_dem_np))
    vmin = 5 * np.floor(all_min / 5)
    vmax = 5 * np.ceil(all_max / 5)
    ims = []
    fig,axes=plt.subplots(1,4,figsize=(16,4),constrained_layout=True)
    ims.append(axes[0].imshow(lr_dem,vmin=vmin,vmax=vmax,cmap="terrain"))
    axes[0].set_title("LR",fontsize=20)
    ims.append(axes[1].imshow(hr_dem,vmin=vmin,vmax=vmax,cmap="terrain"))
    axes[1].set_title("HR",fontsize=20)
    ims.append(axes[2].imshow(bic_dem,vmin=vmin,vmax=vmax,cmap="terrain"))
    axes[2].set_title("Bicubic",fontsize=20)
    ims.append(axes[3].imshow(sr_dem_np,vmin=vmin,vmax=vmax,cmap="terrain"))
    axes[3].set_title("SR",fontsize=20)
    for ax in axes.flatten():
        ax.axis('off')
    # 添加一个 colorbar，挂在最后一个im对象上，也可以直接挂在第一个
    cbar = fig.colorbar(ims[-1], ax=axes, orientation='vertical',shrink=0.7,pad=0.02)
    #cbar.set_label('(m)')
    cbar.ax.set_title('(m)', fontsize=12)
    plt.savefig(os.path.join(output_dir,'LRHRBicSR.svg'),bbox_inches='tight')
    plt.show(block=True)
    #-----------------------------------------#
    # 梯度归一化
    sr_grad_dydx=2/hrsize*dy_dx*1/hr_res
    sr_grad_dx,sr_grad_dy=sr_grad_dydx[...,1],sr_grad_dydx[...,0]
    sr_grad_dx=sr_grad_dx.view(1, 1, hrsize, hrsize).detach().cpu()
    sr_grad_dy=sr_grad_dy.view(1, 1, hrsize, hrsize).detach().cpu()
    sr_grad_dx=sr_grad_dx[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()
    sr_grad_dy=sr_grad_dy[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()

    sr_grad_slope= np.sqrt(sr_grad_dx**2 + sr_grad_dy**2)
    # -----------------------------------------#
    # 差分方法计算
    slope_map = dem_features.Slope_torch(pixel_size=30)
    #sr
    sr_dx,sr_dy,sr_slope=slope_map(sr_4dtensor,return_dxdy=True)
    sr_slope=sr_slope[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()  # 去掉边缘
    sr_dx=sr_dx[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()
    sr_dy=sr_dy[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()
    # hr
    hr_dx,hr_dy,hr_slope=slope_map(hr_4dtensor,return_dxdy=True)
    hr_slope=hr_slope[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()  # 去掉边缘
    hr_dx=hr_dx[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()
    hr_dy=hr_dy[...,1:-1,1:-1].squeeze(0).squeeze(0).numpy()

    # sr_grad_dx_mae= np.mean(np.abs(sr_dx-sr_grad_dx))
    # sr_grad_dy_mae= np.mean(np.abs(sr_dy-sr_grad_dy))
    #
    #
    # print(f'grad result:\n'
    #       f'sr_grad_dx_mae: {sr_grad_dx_mae:.6f},\n'
    #       f'sr_grad_dy_mae: {sr_grad_dy_mae:.6f},\n'
    #       )


    # 坡度可视化
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for ax in axes.flatten():
        ax.axis('off')
    axes[0, 0].imshow(hr_dx, )
    axes[0, 0].set_title('HR dx', fontsize=20)
    axes[0, 1].imshow(hr_dy,)
    axes[0, 1].set_title('HR dy', fontsize=20)
    axes[0, 2].imshow(hr_slope,)
    axes[0, 2].set_title('HR slope', fontsize=20)
    axes[1, 0].imshow(sr_grad_dx, )
    axes[1, 0].set_title('SR dx', fontsize=20)
    axes[1, 1].imshow(sr_grad_dy,)
    axes[1, 1].set_title('SR dy', fontsize=20)
    axes[1, 2].imshow(sr_slope,)
    axes[1, 2].set_title('SR slope', fontsize=20)
    plt.savefig(os.path.join(output_dir,'dxdy.svg'),bbox_inches='tight')
    plt.show(block=True)

    pass
