import sys
import os
import numpy as np
import time
from datetime import datetime
from pathlib import Path
from typing import Union
from natsort import natsorted
import glob
import importlib
import pydoc
from functools import partial

import imageio.v3 as imageio
import logging
import pandas as pd

import torch
import torch.nn.functional as F

import random
import cv2


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


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def arange_inclusive(start,stop,step):
    """
    类似于 np.arange，但确保 stop 被包含在结果中（如果刚好能整除步长）。

    参数:
        start (float): 起始值
        stop (float): 终止值，包含
        step (float): 步长，必须为正数

    返回:
        np.ndarray: 包含右端点的等步长数列
    """
    #arr=np.arange(start,stop+1e-8,step)

    num = int(round((stop - start) / step)) + 1
    return np.linspace(start, start + step * (num - 1), num)


def sort_list_dict(data,sort_key,reverse=False):
    """
    data为字典，每个键对应一个等长列表，按照某个键的列表进行排序，对其他列表对应调整。默认升序
    将字典 data 中所有的列表，按照 data[sort_key] 的值排序后同步重排。
    参数:
        data: Dict[Any, List[Any]]
            原始字典，每个键对应一个等长列表。
        sort_key: Any
            用于排序的键，必须是 data 的一个键。
        reverse: bool
            是否降序排序（True 表示降序，默认升序）。

    返回:
        sorted_data: Dict[Any, List[Any]]
            新字典，和 data 结构一致，但所有列表都已按 sort_key 排序完毕。
    """
    # 1. 检查
    if sort_key not in data:
        raise KeyError(f"sort_key `{sort_key}` 不在字典中")
    N = len(data[sort_key])
    for k, lst in data.items():
        if len(lst) != N:
            raise ValueError(f"所有列表长度必须相同，`{k}` 的长度不为 {N}")

    # 2. 转成 NumPy 数组，方便用 argsort
    arrs = {k: np.array(v) for k, v in data.items()}

    # 3. 计算排序索引
    idxs = np.argsort(arrs[sort_key])
    if reverse:
        idxs = idxs[::-1]

    # 4. 用 fancy indexing 重排，并转回 Python 列表
    sorted_data = {k: arrs[k][idxs].tolist() for k in data}
    return sorted_data


def sort_dict(dict):
    sorted_keys = natsorted(dict.keys())
    sorted_dict = {key: dict[key] for key in sorted_keys}
    return sorted_dict


class Timer():
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        #self.tik = time.time()
        self.tik = time.perf_counter()

    # stop之后需要start
    def stop(self):
        #self.times.append(time.time() - self.tik)
        self.times.append(time.perf_counter() - self.tik)
        return self.times[-1]
    def get_last(self):
        return self.times[-1] if self.times else 0
    def avg(self):
        return sum(self.times) / len(self.times) if self.times else 0

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

    def reset(self):
        self.times.clear()


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)

def second_to_min_sec(seconds):
    """
    将秒转换为分钟和秒的格式
    :param seconds: 输入的秒数
    :return: str 格式为 "mm:ss"
    """
    minutes = int(seconds // 60)  # 获取分钟数
    seconds = int(seconds % 60)  # 获取秒数
    return f"{minutes:02}:{seconds:02}"  # 格式化为 "mm:ss" 的字符串



def compose_multi_avg(epoch_avg):
    """

    :param epoch_avg: {"psnr":AverageMeter(),"ssim":AverageMeter()}
    :return: string
    """
    str=""
    for key,avg_meter in epoch_avg.items():
        str+=f"{key}={epoch_avg[key].avg:.6f}\t"
    return str

#多个变量的累加器
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    #传入参数数量与变量个数一致
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]




image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')



def is_image_file(filename):
    #return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.bmp'])
    return filename.lower().endswith(image_extensions)


def get_image_paths_v1(imagedir):
    # only get image paths in input directory
    img_list = natsorted([os.path.join(imagedir, x) for x in os.listdir(imagedir) if is_image_file(x)])
    return img_list

# slower than v1
def get_image_paths_all_v2(image_dir):
    folder_path = Path(image_dir)
    #recursive glob match all files and directories
    image_files = [str(file) for file in folder_path.rglob('*') if file.suffix.lower() in image_extensions]
    return natsorted(image_files)

def get_image_paths_all_v1(image_dir):
    """
    get all image paths in a directory, including subdirectories.
    :param image_dir:
    :return:
    """
    image_files = []
    # using os.walk to traverse all subdirectories
    for root, dirs, files in os.walk(image_dir):
        for file in files:

            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(root, file))
    return natsorted(image_files)


def get_foldername(path):
    folder_name = os.path.basename(os.path.normpath(path))
    return folder_name
def get_dirname(filepath):

    return os.path.dirname(filepath)


#
# glob模块支持的通配符
# *: 匹配0个或多个字符（不包括目录分隔符）
# ?: 匹配任意单个字符
# [seq]: 匹配seq中的任意字符
# [!seq]: 匹配不在seq中的任意字符
# **: 匹配所有目录和子目录（仅当recursive=True时有效）
def get_files_by_pattern(directory, pattern):
    """
    获取指定目录下匹配特定模式的所有文件路径
    参数:
        directory (str): 要搜索的目录路径
        pattern (str): 文件名匹配模式，支持glob通配符，如'*.txt', 'data_*.csv'等
    返回:
        list: 匹配模式的文件路径列表，按自然排序
    """
    search_pattern = os.path.join(directory, pattern)
    file_paths = glob.glob(search_pattern)
    return natsorted(file_paths)

def get_files_by_pattern_recursive(directory, pattern):
    """
    递归获取指定目录及其子目录下匹配特定模式的所有文件路径
    参数:
        directory (str): 要搜索的目录路径
        pattern (str): 文件名匹配模式，支持glob通配符，如'*.txt', 'data_*.csv'等
    返回:
        list: 匹配模式的文件路径列表，按自然排序
    """
    search_pattern = os.path.join(directory, '**', pattern)
    file_paths = glob.glob(search_pattern, recursive=True)
    return natsorted(file_paths)


def get_nested_cfgstr(cfg_dict:Union[dict,list],indent_num=0)-> str:
    """
    my simple implementation of the codes: OmegaConf.to_yaml(cfg_dict)
    :param cfg_dict: python dict or list from OmegaConf.to_container(cfg,resolve=True) cfg is DictConfig class
    :param indent_num: initial indent numbers
    :return: str
    """
    str=""
    if isinstance(cfg_dict,dict):
        for key, value in cfg_dict.items():
            str+=" "*indent_num+f"'{key}': "
            if isinstance(value, dict):
                str+="\n"
                str+=get_nested_cfgstr(value,indent_num+4)
            elif isinstance(value, list):
                str+="\n"
                str+=get_nested_cfgstr(value,indent_num+2)
            else:
                str+=f"{value}\n"
    if isinstance(cfg_dict, list):
        for item in cfg_dict:
            str += " " * indent_num + "- "
            if isinstance(item, dict):
                str += "\n"
                str+=get_nested_cfgstr(item,indent_num+2)
            elif isinstance(item, list):
                str += "\n"
                str+=get_nested_cfgstr(item,indent_num+2)
            else:
                str+=f"{item}\n"
    return str
def data_equal(a,b,epison=1e-6):
    import torch
    if (type(a)!=type(b)):
        raise TypeError(f"类型不匹配: {type(a).__name__} 与 {type(b).__name__}")
    if isinstance(a,np.ndarray):
        return np.allclose(a,b,atol=epison)
    elif isinstance(a,torch.Tensor):
        return torch.allclose(a,b,atol=epison)
    else:
        raise NotImplementedError


def check_device(model):
    device = next(model.parameters()).device
    print(f"model is on device: {device}")
    return device



# 获取路径下的全部checkpoint，返回间隔epoch_inter的checkpoint列表
# checkpoint命名格式为 model_{epoch}_{loss}.pth
def get_ckpts(dir,epoch_inter=100):
    files=os.listdir(dir)
    result=[]
    for filename in files:
        if filename.endswith('.pth'):
            epoch_str=filename.split('.')[0].split('_')[1]
            try:
                epoch=int(epoch_str)
                if epoch % epoch_inter == 0:
                    file_path=os.path.join(dir,filename)
                    result.append(file_path)
            except ValueError:
                print(f'Invalid filename: {filename}')
    return result


def get_project_root():
    """
    返回项目根目录的绝对路径。
    **本文件**需要被放在项目根目录下的某个子目录里。
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def get_path_from_root(*relative_path_parts):
    """
    从项目根目录构造路径，自动适配系统路径分隔符。
    参数可以是多个字符串，像 os.path.join 一样拼接。
    例：get_path_from_root("models", "my_model.pth")
    """
    root = get_project_root()
    root = Path(root)          # 确保是 Path 类型
    return str(root.joinpath(*relative_path_parts))


def windows_to_linux_path(windows_path: str) -> str:
    """
    将 Windows 路径转换为 Linux 风格路径（适用于标准转换或 WSL 场景）
    示例：
    windows_to_linux_path("Users\\MyName\\data\\file.txt")
    => "Users/MyName/data/file.txt"
    """
    # 替换反斜杠为正斜杠
    path = windows_path.replace("\\", "/")
    return path


def get_object_path(object):
    return object.__module__ + '.' + object.__class__.__name__

def get_object_from_path_v1(class_path):
    module_path, object_name = class_path.rsplit(".", 1)  # 从右边拆成模块名和类名
    try:
        module = importlib.import_module(module_path)             # 动态导入模块
    except ImportError as e:
        raise ImportError(f"{e} 无法导入模块 {module_path}")
    try:
        obj= getattr(module, object_name)
    except AttributeError as e:
        raise AttributeError(f"{e} 模块 {module_path} 中不存在 {object_name} 的对象")

    return obj

def get_object_from_path_v2(object_path):
    object= pydoc.locate(object_path)
    if object is None:
        raise ImportError(f"无法导入对象 {object_path}")
    return object

def instantiate_object(class_path,*args,**kwargs):
    object=get_object_from_path_v1(class_path)

    # 判断获取的对象是否为类
    if not isinstance(object, type):
        raise TypeError(f"{class_path} 不是一个类，而是 {type(object)}")
    try:
        # 实例化类并返回对象
        instance = object(*args, **kwargs)
        return instance
    except Exception as e:
        raise TypeError(f"实例化类 {class_path} 失败: {e}")

def partial_function(func,*args,**kwargs):
    if isinstance(func,str):
        func=get_object_from_path_v1(func)

    if not callable(func):
        raise TypeError(f"{func}不是可调用对象")
    try:
        #
        func_partial = partial(func, *args, **kwargs)
        return func_partial
    except Exception as e:
        raise TypeError(f"partial 失败: {e}")

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

    # filename = r'1.tif'
    # print(is_image_file(filename))
    #raise ShowError("adsdasdasasd")
    # file_path=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Train'
    # res=get_image_paths_all_v1(file_path)


    # dir=r'D:\Data\DEM_data'
    # pattern='*2m.dem'
    # res=get_files_by_pattern_recursive(dir,pattern)

    # data = {
    #     'a': [3, 1, 2],
    #     'b': ['apple', 'banana', 'cherry'],
    #     'c': [30.0, 10.0, 20.0],}
    #
    # # 按 'a' 升序排序
    # asc = sort_list_dict(data, sort_key='a')
    # print("升序 →", asc)
    # # 输出: {'a': [1,2,3], 'b': ['banana','cherry','apple'], 'c': [10.0,20.0,30.0]}
    #
    # # 按 'c' 降序排序
    # desc = sort_list_dict(data, sort_key='a', reverse=True)
    # print("降序 →", desc)
    #test=r'D:\codes\diffusion_dem_sr\weight_file\official_unet.pth'
    #print(windows_to_linux_path(test))

    # str1=r'123\123'
    # str2='123\\123'
    # str11=str1.replace('\\','/')
    # str22=str2.replace('\\','/')

    output_path=get_path_from_root('output/ResDiff/test_server')
    print(output_path)
    # make_dir(output_path)
    pass
