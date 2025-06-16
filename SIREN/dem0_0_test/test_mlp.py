import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from SIREN import meta_modules,dataset,modules
import record
import utils
import dem_data_convert
import dem_features
from utils import to_pixel_samples_tensor,torch_imresize,gradient
from dem_features import Slope_torch,cal_DEM_metric

import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 提前初始化 CUDA 上下文
    torch.cuda.init()

    recorder=record.Recorder(logger=logging.getLogger(__name__),
                             output_path=r'D:\codes\cdem\output\SIREN\test',
                             use_tensorboard=False,
                             extra_name="test_mlp",
                             outfile=False,
                             )
    logger=recorder.get_logger()


    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    weight_path=r"D:\codes\cdem\output\SIREN\test\2025-06-11_19-43-24\mlp_params\dem0_0_mlp_params.pth"
    model_config=record.get_config_from_yaml(r'D:\codes\cdem\output\SIREN\test\2025-06-11_19-43-24\config.yaml')['model_config']
    config={
        "weight_path": weight_path,
    }
    config["model_config"]=model_config
    #recorder.save_config_to_yaml(config)

    mlp_hidden_features = model_config['target_hidden']
    mlp_hidden_layers = model_config['target_hidden_layers']
    image_resolution = model_config['image_resolution']
    use_pe=model_config['use_pe']

    model=modules.SimpleMLPNet(hidden_features=mlp_hidden_features,
                               num_hidden_layers=mlp_hidden_layers,
                               image_resolution=image_resolution,
                               use_pe=use_pe,)
    sd=torch.load(weight_path,weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    hr_height=64
    hr_width=64
    scale=4
    hr_coord = utils.get_pixel_center_coord_tensor((hr_height, hr_width))
    hr_coord = hr_coord.to(device).unsqueeze(0)  # Add batch dimension


    lr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m\dem_0\dem0_0.TIF'
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    #高分低分张量
    hr_dem = utils.read_dem(hr_file_path)
    hr_4dtensor = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)

    lr_dem = utils.read_dem(lr_file_path)
    lr_4dtensor = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    # bic张量
    bic_4dtensor = F.interpolate(lr_4dtensor, scale_factor=scale, mode='bicubic')

    # bic评估结果
    bic_eval_res=cal_DEM_metric(bic_4dtensor,hr_4dtensor,padding=1)

    bic_eval_str=record.compose_kwargs(**bic_eval_res)
    print(f"bicubic dem: {bic_eval_str}")
    # 用于可视化
    slope_map = Slope_torch(pixel_size=30)
    hr_dx,hr_dy,hr_slope=slope_map(hr_4dtensor,return_dxdy=True)
    hr_slope=hr_slope[...,1:-1,1:-1]  # 去掉边缘
    hr_dx=hr_dx[...,1:-1,1:-1]
    hr_dy=hr_dy[...,1:-1,1:-1]

    bic_dx,bic_dy,bic_slope=slope_map.forward(bic_4dtensor,return_dxdy=True)
    bic_slope=bic_slope[...,1:-1,1:-1]
    bic_dx=bic_dx[...,1:-1,1:-1]
    bic_dy=bic_dy[...,1:-1,1:-1]


    # 模型运算
    out=model(hr_coord)
    coord_grad=out['model_in']
    model_out=out['model_out']

    _, trans = dem_data_convert.tensor_maxmin_norm(lr_4dtensor, (-1, 1), 1e-6,
                                                       None)
    sr_value = dataset.value_denorm(model_out, trans)

    # sr
    sr_4dtensor = sr_value.view(1, 1, hr_height, hr_width).detach().cpu()
    # # 保存
    # sr_dem_np=sr_4dtensor.squeeze().squeeze().numpy()
    # utils.write_dem(sr_dem_np,"sr.TIF")

    # sr 评估结果
    sr_eval_res=cal_DEM_metric(sr_4dtensor,hr_4dtensor,padding=1)
    sr_eval_str=record.compose_kwargs(**sr_eval_res)
    print(f"sr dem: {sr_eval_str}")

    # 1,H*W,2
    # dx,dy
    # 自动求导计算得到的偏导数
    sr_grad_dydx= gradient(sr_value, coord_grad, grad_outputs=torch.ones_like(model_out))
    # 2/64 坐标缩放
    sr_grad_dydx=2/64*sr_grad_dydx*1/30
    sr_grad_dx,sr_grad_dy=sr_grad_dydx[...,1],sr_grad_dydx[...,0]
    sr_grad_dx=sr_grad_dx.view(1, 1, hr_height, hr_width).detach().cpu()
    sr_grad_dy=sr_grad_dy.view(1, 1, hr_height, hr_width).detach().cpu()
    sr_grad_dx=sr_grad_dx[...,1:-1,1:-1]
    sr_grad_dy=sr_grad_dy[...,1:-1,1:-1]


    # 栅格计算的梯度
    sr_dem_dx,sr_dem_dy,sr_dem_slope=slope_map.forward(sr_4dtensor,return_dxdy=True)
    sr_dem_slope=sr_dem_slope[...,1:-1,1:-1]
    sr_dem_dx=sr_dem_dx[...,1:-1,1:-1]
    sr_dem_dy=sr_dem_dy[...,1:-1,1:-1]

    sr_grad_dx_mae= torch.mean(torch.abs(sr_grad_dx - sr_dem_dx))
    sr_grad_dy_mae= torch.mean(torch.abs(sr_grad_dy - sr_dem_dy))
    sr_grad_dx_rmse= torch.sqrt(torch.mean(torch.pow(sr_grad_dx - sr_dem_dx, 2)))
    sr_grad_dy_rmse= torch.sqrt(torch.mean(torch.pow(sr_grad_dy - sr_dem_dy, 2)))

    print(f'grad result:\n'
          f'sr_grad_dx_mae: {sr_grad_dx_mae:.6f},\n'
          f'sr_grad_dy_mae: {sr_grad_dy_mae:.6f},\n'
          f'sr_grad_dx_rmse: {sr_grad_dx_rmse:.6f},\n'
          f'sr_grad_dy_rmse: {sr_grad_dy_rmse:.6f}')


    fig,axes= plt.subplots(4,3,figsize=(20,15))
    for ax in axes.flatten():
        ax.axis('off')
    axes[0,0].imshow(hr_4dtensor.squeeze(0).squeeze(0).numpy(), cmap='terrain')
    axes[0,0].set_title('HR DEM')
    axes[0,1].imshow(hr_dx.squeeze(0).squeeze(0).numpy())
    axes[0,1].set_title('dx')
    axes[0,2].imshow(hr_dy.squeeze(0).squeeze(0).numpy())
    axes[0,2].set_title('dy')

    axes[1,0].imshow(bic_4dtensor.squeeze(0).squeeze(0).numpy(),cmap='terrain')
    axes[1,0].set_title('BIC DEM')
    axes[1,1].imshow(bic_dx.squeeze(0).squeeze(0).numpy())
    axes[1,1].set_title('BIC dx')
    axes[1,2].imshow(bic_dy.squeeze(0).squeeze(0).numpy())
    axes[1,2].set_title('BIC dy')

    axes[2,0].imshow(sr_4dtensor.squeeze(0).squeeze(0).detach().cpu().numpy(),cmap="terrain")
    axes[2,0].set_title('SR DEM')
    axes[2,1].imshow(sr_dem_dx.squeeze(0).squeeze(0).numpy())
    axes[2,1].set_title('SR dx')
    axes[2,2].imshow(sr_dem_dy.squeeze(0).squeeze(0).numpy())
    axes[2,2].set_title('SR dy')


    axes[3,1].imshow(sr_grad_dx.squeeze(0).squeeze(0).detach().cpu().numpy())
    axes[3,1].set_title('SR grad dx')
    axes[3,2].imshow(sr_grad_dy.squeeze(0).squeeze(0).detach().cpu().numpy())
    axes[3,2].set_title('SR grad dy')

    plt.savefig("dem0_0_dxdy.png")
    plt.show(block=True)




    pass
