import sys
import os


import numpy as np
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_


import utils

from compared_model.tfasr import Net


import time
if __name__ == '__main__':

    utils.seed_everything(utils.default_seed)
    device=utils.default_device

    lrsize=16
    scale=4
    hrsize=lrsize*scale


    tfasr_config={
        "n_residual_blocks": 16,
        "upsample_factor": scale,
    }

    # 模型，优化器，调度器
    net=Net(**tfasr_config)
    net.to(device)

    net.eval()

    dummy_input = torch.randn(1, 1, 16, 16).cuda()
    dummy_output = F.interpolate(dummy_input, scale_factor=scale, mode='nearest')

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

