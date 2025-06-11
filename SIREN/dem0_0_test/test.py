import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from SIREN import meta_modules,dataset,modules
from utils import to_pixel_samples_tensor,torch_imresize
import utils
import dem_data_convert
import dem_features

import matplotlib.pyplot as plt

# 提前初始化 CUDA 上下文
torch.cuda.init()

# 测试单个dem文件并可视化其梯度

if __name__ == '__main__':
    pass
