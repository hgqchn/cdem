import sys
import os


import numpy as np
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


import utils


from SIREN import meta_modules,modules

from SIREN.dataset import DEMFolder, ImplicitDownsampled,value_denorm
from mymodel.model import ImplicitModel
from mymodel.SwinIR import SwinEncoder


import time
if __name__ == '__main__':

    utils.seed_everything(utils.default_seed)
    device=utils.default_device

    lrsize=16
    scale=4
    hrsize=lrsize*scale
    config={
        "scale": scale,
        "hrsize": hrsize,
        "content_loss": 1.0,
        "slope_loss": 0,
        "fft_loss": 0,
        "lr_decay_steps": 100,
    }
    laten_code_dim=256
    swin_encoder_config={
        "img_size": 16,
        "patch_size": 1,
        "in_chans": 1,
        "embed_dim": laten_code_dim,
        "depths": [4, 4, 4, 4],
        "num_heads": [4, 4, 4, 4],
        "window_size": 1,
        "mlp_ratio": 4.0,
    }
    target_config={
        "out_features": 1,
        "hidden_features": 16,
        "num_hidden_layers": 2,
        "use_pe": False,
        "num_frequencies": 10,
        "use_hsine": False,
    }
    hyper_config={
        "hyper_in_features": laten_code_dim,
        "hyper_hidden_layers": 3,
        "hyper_hidden_features": laten_code_dim,
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


    target_net=modules.SimpleMLPNetv1(**target_config)
    hyper_net=meta_modules.HyperNetwork(**hyper_config,
                                       target_module=target_net)
    encoder=SwinEncoder(**swin_encoder_config)

    net=ImplicitModel(encoder=encoder,
                             hyper_net=hyper_net,
                             target_net=target_net)
    net.to(device)

    net.eval()

    dummy_input = torch.randn(1, 1, 16, 16).cuda()

    hr_coord= utils.get_pixel_center_coord_tensor((64,64))
    hr_coord=hr_coord.unsqueeze(0).to(device)
    # 预热（warm-up）
    for _ in range(20):
        _ = net(dummy_input,hr_coord)

    # 测试推理时间
    torch.cuda.synchronize()

    generate_mlp_time=0
    mlp_time=0
    start = time.perf_counter()
    # 正式计时
    with torch.no_grad():
        for _ in range(50):
            #_ = net(dummy_input,hr_coord)
            generate_mlp_t1=time.perf_counter()
            embedding = net.encoder(dummy_input)
            target_params = net.hyper_net(embedding)
            generate_mlp_t2 = time.perf_counter()
            generate_mlp_time += (generate_mlp_t2 - generate_mlp_t1)
            mlp_t1= time.perf_counter()
            model_output = net.target_net(hr_coord, params=target_params)
            mlp_t2 = time.perf_counter()
            mlp_time += (mlp_t2 - mlp_t1)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_avg_time = (end - start) / 50
    print(f'total_avg_time: {total_avg_time*1000} ms'
          f'\ngenerate_mlp_avg_time: {generate_mlp_time / 50*1000} ms'
          f'\nmlp_avg_time: {mlp_time / 50*1000 } ms')


    pass
