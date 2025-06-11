import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset


import logging
import wandb
from tqdm import tqdm

import copy

import record
import utils

import dem_features
import dem_data_convert

from utils import to_pixel_samples_tensor,torch_imresize
from SIREN import meta_modules,dataset

#


if __name__ == '__main__':
    recorder=record.Recorder(logger=logging.getLogger(__name__),
                             output_path=r'D:\codes\cdem\output\SIREN\test',
                             use_tensorboard=False,
                             )
    logger=recorder.get_logger()


    # mlp权重参数保存地儿
    mlp_params_save_path=os.path.join(recorder.save_path,'mlp_params')
    utils.make_dir(mlp_params_save_path)

    utils.seed_everything(utils.default_seed)
    device=utils.default_device

    model_name="SIREN"
    weight_path=r'D:\codes\cdem\output\SIREN\2025-05-28_10-15-42\checkpoint\best_SIREN_201_2.5209.pth'

    current_time = utils.get_current_time()

    config={
        "weight_path": weight_path,
    }

    model_config=record.get_config_from_yaml(r'D:\codes\cdem\output\SIREN\2025-05-28_10-15-42\config.yaml')['model_config']
    config["model_config"]=model_config

    recorder.save_config_to_yaml(config)

    #-----------------------------------------#
    # data
    lr_file_path= r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m'
    hr_file_path= r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test'

    lr_file_list= utils.get_dem_paths_all(lr_file_path)
    hr_file_list= utils.get_dem_paths_all(hr_file_path)


    #-----------------------------------------#
    # 模型

    model=meta_modules.ConvolutionalNeuralProcessImplicit2DHypernet(
        **model_config
    )
    sd=torch.load(weight_path,weights_only=True)
    model.load_state_dict(sd)
    model.to(device)

    eval_results = {
        'name': [],
        'height_mae': [],
        'height_rmse': [],
        "bic_height_mae": [],
        "bic_height_rmse": [],
        'slope_mae': [],
        'slope_rmse': [],
        'aspect_mae': [],
        'aspect_rmse': [],
    }

    # 测试
    model.eval()


    with tqdm(total=len(lr_file_list), desc=f' test', file=sys.stdout) as t:
        for lr,hr in zip(lr_file_list, hr_file_list):

            assert utils.get_filename(lr,with_ext=False) == utils.get_filename(hr,with_ext=False)
            filename=utils.get_filename(lr,with_ext=False)
            eval_results["name"].append(filename)
            hr_dem=utils.read_dem(hr)
            hr_height,hr_width=hr_dem.shape
            hr_dem=torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)

            lr_dem=utils.read_dem(lr)
            # 1,1,H,W
            lr_dem = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
            scale=int(hr_dem.shape[2]/lr_dem.shape[2])

            bic_dem=F.interpolate(lr_dem,scale_factor=scale,mode='bicubic')

            hr_coord=utils.get_pixel_center_coord_tensor((hr_height,hr_width))
            # 1,H*W,2
            hr_coord=hr_coord.unsqueeze(0)


            input, trans = dem_data_convert.tensor_maxmin_norm(lr_dem, (-1, 1), 1e-6,
                                                                   None)

            input=input.to(device)
            hr_coord=hr_coord.to(device)

            trans=trans.to(device)


            with torch.inference_mode():
                model_out = model(input, hr_coord)
                # 1,H*W,1
                sr_value = model_out['model_out']
                mlp_params=model_out['hypo_params']

                torch.save(mlp_params, os.path.join(mlp_params_save_path, f'{filename}_params.pth'))

            # 1,H*W,1
            sr_value= dataset.value_denorm(sr_value,trans)

            sr_dem=sr_value.view(1,1,hr_height,hr_width)

            eval_res=dem_features.cal_DEM_metric(sr_dem,hr_dem,device=device)
            eval_res_bic=dem_features.cal_DEM_metric(bic_dem,hr_dem,device=device)
            for key, value in eval_res.items():
                eval_results[key].append(value)
            eval_results["bic_height_mae"].append(eval_res_bic['height_mae'])
            eval_results["bic_height_rmse"].append(eval_res_bic['height_rmse'])

            a=1


    pass
