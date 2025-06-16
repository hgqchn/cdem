import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

from SIREN import meta_modules,dataset,modules
from utils import to_pixel_samples_tensor,torch_imresize

import record
import utils
import dem_data_convert
import dem_features

import matplotlib.pyplot as plt

# 提前初始化 CUDA 上下文
torch.cuda.init()

# 测试单个dem文件并可视化其梯度


if __name__ == '__main__':

    recorder=record.Recorder(logger=logging.getLogger(__name__),
                             output_path=r'D:\codes\cdem\output\SIREN\test',
                             use_tensorboard=False,
                             )
    logger=recorder.get_logger()

    # test data
    lr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m\dem_0\dem0_0.TIF'
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    filename = utils.get_filename(lr_file_path, with_ext=False)
    scale=4
    hr_dem = utils.read_dem(hr_file_path)
    hr_4Dtensor = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)
    lr_dem = utils.read_dem(lr_file_path)
    lr_4Dtensor = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    bic_4Dtensor = F.interpolate(lr_4Dtensor, scale_factor=scale, mode='bicubic')

    device = utils.default_device

    weight_path=r'D:\codes\cdem\output\SIREN\2025-06-06_17-11-25 no pe\checkpoint\best_SIREN_152_2.3937.pth'
    model_config=record.get_config_from_yaml(r'D:\codes\cdem\output\SIREN\2025-06-06_17-11-25 no pe\config.yaml')['model_config']
    config={
        "weight_path": weight_path,
    }
    config["model_config"]=model_config

    recorder.save_config_to_yaml(config)


    # mlp权重参数保存地儿
    mlp_params_save_path=os.path.join(recorder.save_path,'mlp_params')
    utils.make_dir(mlp_params_save_path)

    # 模型

    model=meta_modules.ConvolutionalNeuralProcessImplicit2DHypernet(
        **model_config
    )
    sd=torch.load(weight_path,weights_only=True)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    hr_height, hr_width = hr_dem.shape
    hr_coord = utils.get_pixel_center_coord_tensor((hr_height, hr_width))
    # 1,H*W,2
    hr_coord = hr_coord.unsqueeze(0)
    hr_coord = hr_coord.to(device)

    input, trans = dem_data_convert.tensor_maxmin_norm(lr_4Dtensor, (-1, 1), 1e-6,None)
    input = input.to(device)
    trans = trans.to(device)

    with torch.inference_mode():
        model_out = model(input, hr_coord)
        # 1,H*W,1
        sr_value = model_out['model_out']
        mlp_params = model_out['hypo_params']
        for name,params in mlp_params.items():
            mlp_params[name].squeeze_(0)
        torch.save(mlp_params, os.path.join(mlp_params_save_path, f'{filename}_mlp_params.pth'))


    pass
