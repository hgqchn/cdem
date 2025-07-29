import sys
import os

import re
import numpy as np
import torch
import torch.nn.functional as F

import record
import utils
import dem_features

import matplotlib.pyplot as plt

from SIREN import modules

from SIREN.dataset import value_denorm


if __name__ == '__main__':

    device = utils.default_device

    current_time = utils.get_current_time()

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


    pass
