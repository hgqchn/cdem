import sys
import os


import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


import utils

from SIREN.dataset import DEMFolder, ImplicitDownsampled,value_denorm

from mymodel.SwinIR import SwinIR


import time
if __name__ == '__main__':

    utils.seed_everything(utils.default_seed)
    device=utils.default_device

    lrsize=16
    scale=4
    hrsize=lrsize*scale

    laten_code_dim=256
    swin_config={
        "img_size": 16,
        "patch_size": 1,
        "in_chans": 1,
        "embed_dim": laten_code_dim,
        "depths": [4, 4, 4, 4],
        "num_heads": [4, 4, 4, 4],
        "window_size": 1,
        "mlp_ratio": 4.0,
        "upsampler":"pixelshuffle",
        "upscale": scale,
    }

    #-----------------------------------------#
    # data
    train_folder=DEMFolder(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Train')
    train_dataset=ImplicitDownsampled(
        dataset=train_folder,
        scale=scale,
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=32,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=4)


    test_folder=DEMFolder(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test',
                          )
    test_dataset=ImplicitDownsampled(
        dataset=test_folder,
        scale=scale,
    )
    test_loader = DataLoader(test_dataset,
                            batch_size=32,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              num_workers=4)
    #-----------------------------------------#
    # 模型，优化器，调度器
    net=SwinIR(**swin_config)
    net.to(device)

    net.eval()

    dummy_input = torch.randn(1, 1, 16, 16).cuda()


    # 预热（warm-up）
    for _ in range(20):
        _ = net(dummy_input)

    # 测试推理时间
    torch.cuda.synchronize()

    start = time.perf_counter()
    # 正式计时
    with torch.no_grad():
        for _ in range(50):
            out = net(dummy_input)

    torch.cuda.synchronize()
    end = time.perf_counter()

    total_avg_time = (end - start) / 50
    print(f'total_avg_time: {total_avg_time*1000} ms')


    pass

