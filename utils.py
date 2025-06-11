import sys
import os
import numpy as np
import imageio.v3 as imageio
import logging
import pandas as pd

from natsort import natsorted

import torch
import torch.nn.functional as F

import random
import cv2

from datetime import datetime
import time

torch.set_printoptions(precision=6)

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

default_seed=3407

default_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

default_dem_norm_range=(-1,1)
default_dem_norm_eps=10



def seed_everything(seed,reproducibility=True):
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    random.seed(seed)
    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现

    if reproducibility:
        torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
        torch.backends.cudnn.deterministic = True  # 选择确定性算法
    else:
        torch.backends.cudnn.benchmark = True  # if benchmark=True, deterministic will be False
        #torch.backends.cudnn.deterministic = True  # 选择确定性算法

def read_dem(dem_file: str):
    """

    :param dem_file:
    :return: H W
    """
    file_suffix = dem_file.split('.')[-1]
    if file_suffix.lower() in ['tif', 'tiff']:
        # 和pillow库速度差不多
        # dem_data = np.array(Image.open(dem_file))
        dem_data = imageio.imread(dem_file)
    elif file_suffix == 'dem':
        # 后缀是dem的文件，其实就是ASCII文件，用读取txt的方式读取
        dem_data = np.loadtxt(dem_file, dtype=np.float32, delimiter=',')
    else:
        raise ValueError(f"Unsupported file format: {file_suffix}")
    return dem_data.astype(np.float32)

def write_dem(dem_data, dem_file):
    imageio.imwrite(dem_file, dem_data)

def custom_logger(logger_name=__name__,log_filepath=None):
    logger=logging.getLogger(logger_name)
    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(filename)s][%(levelname)s] - %(message)s")
        logger.setLevel(logging.INFO)
        if log_filepath:
            filehandler = logging.FileHandler(log_filepath, encoding="utf-8")
            filehandler.setFormatter(formatter)
            filehandler.setLevel(logging.INFO)
            logger.addHandler(filehandler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    return logger

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def make_dir(path):
    os.makedirs(path,exist_ok=True)

dem_extensions=('.tif', '.tiff', '.dem')
def is_dem_file(filename):
    return filename.lower().endswith(dem_extensions)

def get_dem_paths(dir):
    # only get image paths in input directory
    img_list = natsorted([os.path.join(dir, x) for x in os.listdir(dir) if is_dem_file(x)])
    return img_list

def get_dem_paths_all(dir):
    """
    get all image paths in a directory, including subdirectories.
    """
    dem_files = []
    # using os.walk to traverse all subdirectories
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.lower().endswith(dem_extensions):
                dem_files.append(os.path.join(root, file))
    return natsorted(dem_files)

def get_filename(path, with_ext=False):
    filename = os.path.basename(path)
    if with_ext:
        return os.path.splitext(filename)
    else:
        return os.path.splitext(filename)[0]

def get_parameter_nums(model,show=True):

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total_params - trainable_params
    #print('#parameters:', nums)
    if show:
        print(f"\nModel: {model.__class__.__name__}")
        print(f"Total parameters:      {total_params:,}")
        print(f"Trainable parameters:  {trainable_params:,}")
        print(f"Non-trainable params:  {non_trainable:,}\n")

    return total_params,trainable_params,non_trainable

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()  # 元素个数 × 每个元素占字节数
    return param_size / 1024 / 1024  # 单位转换为 MB



def cv2_imresize(img, scale, mode="bicubic"):
    """

    :param img: numpy array shape must be divided by scale
    :param scale:
    :param mode:
    :return:
    """
    if mode=="bicubic":
        cv2_mode=cv2.INTER_CUBIC
    elif mode=="bilinear":
        cv2_mode=cv2.INTER_LINEAR
    elif mode=="nearest":
        cv2_mode=cv2.INTER_NEAREST
    elif mode=="lanczos":
        cv2_mode=cv2.INTER_LANCZOS4
    else:
        raise NotImplementedError('interpolation mode not supported')
    height=int(img.shape[0]*scale)
    width=int(img.shape[1]*scale)

    result=cv2.resize(img,dsize=None,fx=scale,fy=scale,interpolation=cv2_mode)
    return result

def torch_imresize(img, scale=None, shape=None,mode="bicubic", align_corners=False):

    if shape is not None:
        height,width=shape
    elif scale is not None:
        height=int(img.shape[0]*scale)
        width=int(img.shape[1]*scale)
    else:
        raise ValueError("Either scale or shape must be provided")

    if img.ndim==2:
        img=np.expand_dims(img,-1)
    #1,C,H,W
    img_t=torch.from_numpy(img).permute(2,0,1).unsqueeze(0)
    # align_corner=False, results close to cv2
    if mode=="nearest":
        result = F.interpolate(img_t, size=(height, width), mode=mode)
    else:
        result=F.interpolate(img_t, size=(height,width), mode=mode,align_corners=align_corners)
    result_np=result.squeeze(0).permute(1,2,0).numpy().squeeze()
    return result_np

def save_dict_csv_pd(data_dict,save_file="./data.csv"):
    # 元素不能为空
    df=pd.DataFrame(data_dict)
    df.to_csv(save_file, index=False)
    return os.path.abspath(save_file)

def compose_kwargs(**kwargs):
    """
    compose keyword arguments to a string
    :param kwargs:
    :return:
    """
    str_res=""
    for key, value in kwargs.items():
            str_res=str_res+(f"{key}:{str(value)}")+" "*4
    return str_res

def get_current_time():
    return datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# 根据输入的形状和范围（默认-1,1）
# shape=(H,W)
# 生成网格坐标 (纵坐标，横坐标) 对应于H，W
def get_pixel_center_coord_tensor(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
        ranges:2*2大小的 range[0]=() range[1]=() 网格边界的值
        生成网格中心点对应的坐标
    """
    coord_seqs = []
    # shape = [H, W]
    # i=0,n=H
    # i=1,n=W

    shape= (shape,shape) if isinstance(shape, int) else shape

    # shape H,W
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]

        # 0.5 pixel width
        r = (v1 - v0) / (2 * n)
        # seq范围[vo+r,v1-r]，n个数，间隔2r
        # torch.arange(n) 0~n-1
        # (v0+r,v0+3r,v0+5r,...,v1-r)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        # 存储H 和 W的坐标
        coord_seqs.append(seq)

    # *coord_seqs 是H W
    # 输入是先纵轴坐标后横轴坐标
    # 指定index为ij形式，则生成的两个网格坐标数组大小是H W，对应列索引和行索引,取值就是[H][W]。
    # stack后，大小变为H W 2  2中首先是对应的H坐标然后是W坐标
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    # 数组展平(H*W,2)默认为True
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

def to_pixel_samples_tensor(img):
    """ Convert the image to coord-value pairs.
        img: Tensor, (C, H, W)
    """
    # img.shape[-2:] = (H, W)
    # coord:
    coord = get_pixel_center_coord_tensor(img.shape[-2:])
    # 通道 1
    c = img.shape[0]
    # 先将img变为C H*W，再调换顺序变为H*W C
    img_value = img.view(c, -1).permute(1, 0)  # (H*W, C)
    # coord: H*W,2
    # img_value: H*W,C C=1
    return coord, img_value

def get_point_coord_tensor(sidelen, flatten=True):
    """
    Make coordinates at grid points.
    grid lenth=2 coord number=3
    :param sidelen: int or tuple, length of the side of the grid, e.g. 3
    :param flatten: bool, whether to flatten the output tensor
    :return pixel_coords: Tensor, shape (H*W, 2) if flatten is True, otherwise (H, W, 2)
    """
    if isinstance(sidelen, int):
        sidelen = 2 * (sidelen,)
    # sidelen=(H,W)
    # mgrid 先y后x 相当于meshgrid(np.arange(H), np.arange(W),indexing='ij')
    pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1).astype(np.float32)
    pixel_coords[:, :, 0] = pixel_coords[:, :, 0] / (sidelen[0] - 1)
    pixel_coords[:, :, 1] = pixel_coords[:, :, 1] / (sidelen[1] - 1)


    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.tensor(pixel_coords)
    if flatten:
        pixel_coords = pixel_coords.view(-1, 2)  # (H*W, 2)
    return pixel_coords

def gradient(y, x, grad_outputs=None):
    """
    Calculate the gradient of y with respect to x.
    :param y:
    :param x:
    :param grad_outputs:
    :return: shape same as x, gradient of y with respect to x
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad



if __name__ == '__main__':
    #print(__name__)
    # logger=custom_logger(__name__)
    # logger.info("test")
    # print(get_current_time())
    shape=(3,4)
    res1=get_pixel_center_coord_tensor(shape)
    res2=get_point_coord_tensor(shape)

    #
    np_res1=np.stack(np.mgrid[:3, :4], axis=-1)
    np_res2=np.stack(np.meshgrid(np.arange(3),np.arange(4),indexing='ij'), axis=-1)
    pass
