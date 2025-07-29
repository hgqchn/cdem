import sys
import os


import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from torch.nn.utils import clip_grad_norm_

import logging
import wandb
from tqdm import tqdm

import copy

import record
import utils
import dem_features
import dem_data_convert
from SIREN import meta_modules,modules

from SIREN.dataset import DEMFolder, ImplicitDownsampled,value_denorm

import time

# 一次性读取图像所有像素点的坐标和值
class DEMFitting(Dataset):
    def __init__(self, dem):
        super().__init__()
        assert dem.ndim == 2
        assert dem.shape[-1] == dem.shape[-2]

        # C,H,W
        dem_t = torch.from_numpy(dem).unsqueeze(0)
        data_norm, trans = dem_data_convert.tensor_maxmin_norm(dem_t)
        self.trans=trans

        self.pixels = data_norm.permute(1, 2, 0).view(-1, 1)
        self.coords = utils.get_pixel_center_coord_tensor(dem.shape[0])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels,self.trans

if __name__ == '__main__':

    utils.seed_everything(utils.default_seed)
    device=utils.default_device

    mlp_config={
        "out_features": 1,
        "hidden_features": 16,
        "num_hidden_layers": 2,
        "use_pe": False,
        "num_frequencies": 10,
        "use_hsine": False,
    }


    #-----------------------------------------#
    # data
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    hr_dem = utils.read_dem(hr_file_path)
    dataset=DEMFitting(hr_dem)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)

    #-----------------------------------------#

    model = modules.SimpleMLPNetv1(**mlp_config)
    model.to(device)
    model.train()

    total_steps = 2000  # Since the whole image is our dataset, this just means 500 gradient descent steps.

    test_step=100
    test_flag=True

    optimizer = torch.optim.Adam(lr=1e-4, params=model.parameters())
    # 1,65536,2  1,65536,1
    model_input, ground_truth,trans = next(iter(dataloader))
    trans=trans.to(device)
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    total_time=0
    for step in range(1,total_steps+1):
        start_time = time.perf_counter()
        # 一次性把所有像素值输入
        model_out= model(model_input)
        model_output=model_out['model_out']
        coords=model_out['model_in']

        #model_output, coords = model_out

        loss = ((model_output - ground_truth) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end_time = time.perf_counter()
        train_time = end_time - start_time
        total_time += train_time
        if test_flag:
            if step % test_step == 0:
                print(f"Step {step}, Loss: {loss.item()}")
                # 测试模型输出
                model.eval()
                with torch.no_grad():
                    model_out = model(model_input)
                    model_output = model_out['model_out']
                    model_fit_denorm = value_denorm(model_output, trans)
                    ground_truth_denorm = value_denorm(ground_truth, trans)

                    height_rmse = torch.sqrt(torch.mean(torch.pow(model_fit_denorm - ground_truth_denorm, 2)))
                    print(f"Step {step},Height RMSE: {height_rmse}")
                    print(f"{step} fitting time: {total_time} s")


    # print(f"{total_steps} fitting time: {total_time} s")

    model.eval()
    model_out = model(model_input)
    model_output = model_out['model_out']
    model_fit_denorm=value_denorm(model_output,trans)
    ground_truth_denorm=value_denorm(ground_truth,trans)

    height_rmse=torch.sqrt(torch.mean(torch.pow(model_fit_denorm - ground_truth_denorm, 2)))
    print(f"Height RMSE: {height_rmse.item()}")
    pass
