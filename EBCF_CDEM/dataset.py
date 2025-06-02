import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import transforms

import random
import math
import utils
import dem_data_convert
from utils import to_pixel_samples_tensor


def value_denorm(norm_value, trans):

    # value: B,L,1
    # trans: B,3

    # 确保输入张量在同一设备上
    if norm_value.device != trans.device:
        trans = trans.to(norm_value.device)

    # B 3->B 1-> B 1 1
    data_min= trans[:, 0].reshape(-1, 1, 1)
    norm_min= trans[:, 1].reshape(-1, 1, 1)
    scale= trans[:, 2].reshape(-1, 1, 1, 1)
    # B L 1
    new_data= (norm_value - norm_min) / scale + data_min
    return new_data

class DEMFolder(Dataset):
    def __init__(self, dem_dir, norm_range=(-1, 1), epsilon=1e-6,minmax_height=None,
                 repeat=1):
        super(DEMFolder, self).__init__()
        self.dem_list = utils.get_dem_paths_all(dem_dir)
        self.repeat= repeat
        self.norm_range = norm_range
        self.epsilon = epsilon

        # 使用传入的min max值进行归一化，而不是每个dem分别maxmin归一化
        self.minmax_height=minmax_height


    def __getitem__(self, index):
        dem_file=self.dem_list[index%len(self.dem_list)]
        filename = utils.get_filename(dem_file)

        data=utils.read_dem(dem_file)

        # 64 64 -> 1 64 64
        data_t = torch.from_numpy(data).unsqueeze(0).float()
        data_norm, trans = dem_data_convert.tensor_maxmin_norm(data_t, self.norm_range, self.epsilon,self.minmax_height)

        # data_norm 1 64 64
        # trans 3
        return data_norm,trans

    def __len__(self):
        return len(self.dem_list)*self.repeat


class SDFImplicitDownsampled(Dataset):
    def __init__(self,
                 dataset, #DEMFolder 类
                 image_size=16,
                 scale_min=1,
                 scale_max=4,
                 sample_q=None,
                 ) -> None:
        '''
        Distribute the elevation value at (x,y)
        image_size 和 sample_q需要在训练时指定，确定batch内一致
        '''
        super().__init__()
        self.dataset = dataset
        self.length = len(dataset)

        self.image_size = image_size
        self.sample_q = sample_q  # 256
        self.scale_min = scale_min #1

        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max #4



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # img 1,H,W
        # trans 3
        # img为归一化后的数据，trans保留了归一化参数
        img,trans = self.dataset[idx]

        #坐标，值对
        img_coord, img_value =utils.to_pixel_samples_tensor(img.contiguous())
        #H*W,2-> 2,H*W->2,H,W
        img_coord = img_coord.permute(1,0).view([-1, *(img.shape[1:])])

        #随机生成一个数，作为降采样因子/缩放因子
        s = random.uniform(self.scale_min, self.scale_max)

        if self.image_size is None:
            #低分辨率图像的尺寸
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            #裁剪原图像，从左上角裁剪
            #大小变为round(h_lr * s),round(w_lr * s)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            #作者包装的resize函数，调用的transform的函数
            #生成低分辨率图像
            # 1,h_lr,w_lr
            img_down = resize_fn(img, (h_lr, w_lr))

            #裁剪后的低分辨率和高分辨率图像
            lr, hr = img_down, img
        else:
            assert self.image_size <= min(img_coord.shape[-2:])
            lr_size = self.image_size
            hr_size = round(lr_size * s)
            # 裁剪原图像，随机确定裁剪的边界
            #
            x0 = random.randint(0, img.shape[-2] - hr_size)
            y0 = random.randint(0, img.shape[-1] - hr_size)

            # 裁剪后的高分图像
            crop_hr = img[:, x0: x0 + hr_size, y0: y0 + hr_size]
            # 裁剪后的低分图像
            crop_lr = resize_fn(crop_hr, lr_size)

            #裁剪后的低分辨率和高分辨率图像
            lr, hr = crop_lr,crop_hr

        #高分图像对应的坐标，值对
        # hr_coord: H*W,2 范围-1,1
        # hr_value: H*W,1
        hr_coord, hr_value = to_pixel_samples_tensor(hr.contiguous())


        #接着随机采样
        #sample_q表示要选取的元素个数
        #replace是否可以重复选取
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            #高分的坐标，值对，全局坐标
            hr_coord = hr_coord[sample_lst]
            hr_value = hr_value[sample_lst]


        return lr,hr_coord,hr_value,trans



def resize_fn(img, size, method='nearest'):
    if method == 'bicubic':
        interpolation = transforms.InterpolationMode.BICUBIC
    elif method == 'bilinear':
        interpolation = transforms.InterpolationMode.BILINEAR
    elif method == 'nearest':
        interpolation = transforms.InterpolationMode.NEAREST
    else:
        raise Exception('Please align interpolation method')

    return transforms.Resize(size, interpolation)(img)

if __name__ == '__main__':
    pass
