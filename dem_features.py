import numpy as np
import torch
import torch.nn as nn
import math
import cv2
import dem_data_convert
#用于评估指标的计算

class Slope_torch(nn.Module):
    def __init__(self,pixel_size=1):
        #pixel_size在实际计算坡度时使用,为空间分辨率
        super(Slope_torch, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)
        self.pixel_size = pixel_size
        # -1 0 1
        # -2 0 2
        # -1 0 -1
        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1
        # -1 -2 -1
        # 0 0 0
        # 1 2 1
        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8 * self.pixel_size)
        weight2 = weight2 / (8 * self.pixel_size)
        # nn.Parameter 注册为模型参数
        self.weight1 =nn.Parameter(torch.tensor(weight1)) # 自定义的权值
        self.weight2 =nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        self.bias.requires_grad = False
    def forward(self, x,return_dxdy=False):
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        # 坡度值
        slope = torch.arctan(slope) * 180 / math.pi

        if return_dxdy:
            return dx, dy, slope
        else:
            return slope

    def forward_dxdy(self, x):
        """
        计算dx和dy
        :param x: 输入张量
        :return: dx, dy, grad_norm
        """
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        grad_norm= torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))

        return dx, dy,grad_norm



#坡向
class Aspect_torch(nn.Module):
    def __init__(self):
        super(Aspect_torch, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = -1
        weight1[0][1] = 0
        weight1[0][2] = 1
        weight1[1][0] = -2
        weight1[1][1] = 0
        weight1[1][2] = 2
        weight1[2][0] = -1
        weight1[2][1] = 0
        weight1[2][2] = 1

        weight2[0][0] = -1
        weight2[0][1] = -2
        weight2[0][2] = -1
        weight2[1][0] = 0
        weight2[1][1] = 0
        weight2[1][2] = 0
        weight2[2][0] = 1
        weight2[2][1] = 2
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))

        self.weight1 = nn.Parameter(torch.tensor(weight1))  # 自定义的权值
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias =nn.Parameter(torch.zeros(1))  # 自定义的偏置
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        # west point to east
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        # north point to south
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        # torch.atan2(-dy, dx) 返回向量(dx,-dy)与x轴正方向的弧度，范围在-pi到pi，逆时针为正
        aspect = 180/math.pi*torch.atan2(-dy, dx)

        # angle from north
        aspect = torch.where(aspect > 90, 360 - aspect + 90, 90 - aspect)
        return aspect

    def forward_rad(self, x):
        # west point to east
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        # north point to south
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)

        # torch.atan2(-dy, dx) 返回向量(dx,-dy)与x轴正方向的弧度，范围在-pi到pi，逆时针为正
        aspect_rad = torch.atan2(-dy, dx)

        return aspect_rad


Slope_net = Slope_torch(pixel_size=30)
Aspect_net = Aspect_torch()

def cal_DEM_metric(demA, demB, padding=None, device=None, reduction="mean",slope_net=Slope_net, aspect_net=Aspect_net):
    """
    input ndarray or tensor
    :param demA demB: B 1 H W
    :param padding: to be cropped
    :reduction: batch mean or batch res
    :return: {
        'height_mae':
        'slope_mae':
        'aspect_mae':
        'height_rmse':
        'slope_rmse':
        'aspect_rmse':
        }
    """
    B=demA.shape[0]
    if padding:
        demA = demA[...,padding:-padding, padding:-padding]
        demB = demB[...,padding:-padding, padding:-padding]
    if isinstance(demA,np.ndarray):
        demA_tensor = torch.from_numpy(demA)
        demB_tensor = torch.from_numpy(demB)
    elif isinstance(demA, torch.Tensor):
        demA_tensor=demA
        demB_tensor=demB

    slope_net=slope_net
    aspect_net=aspect_net
    if device:
        slope_net.to(device)
        aspect_net.to(device)
        demA_tensor=demA_tensor.to(device)
        demB_tensor=demB_tensor.to(device)
    else:
        device=demA_tensor.device
        slope_net.to(device)
        aspect_net.to(device)
        demB_tensor=demB_tensor.to(device)

    with torch.inference_mode():
        demA_slope = slope_net(demA_tensor)
        demB_slope = slope_net(demB_tensor)
        demA_aspect = aspect_net(demA_tensor)
        demB_aspect = aspect_net(demB_tensor)

    height_mae=torch.abs(demA_tensor - demB_tensor).mean(dim=(1,2,3))
    height_rmse=torch.sqrt(torch.mean(torch.pow(demA_tensor - demB_tensor, 2), dim=(1,2,3)))
    height_max_error,_=torch.abs(demA_tensor - demB_tensor).view(B,-1).max(dim=1)

    slope_mae=torch.abs(demA_slope - demB_slope).mean(dim=(1,2,3))
    slope_rmse=torch.sqrt(torch.mean(torch.pow(demA_slope - demB_slope, 2), dim=(1,2,3)))
    slope_max_error,_=torch.abs(demA_slope - demB_slope).view(B,-1).max(dim=1)

    aspect_mae=torch.abs(demA_aspect - demB_aspect).mean(dim=(1,2,3))
    aspect_rmse=torch.sqrt(torch.mean(torch.pow(demA_aspect - demB_aspect, 2), dim=(1,2,3)))
    aspect_max_error,_=torch.abs(demA_aspect - demB_aspect).view(B,-1).max(dim=1)
    #B -> 1
    if reduction=="mean":
        height_mae=height_mae.mean(dim=0,keepdim=True)
        height_rmse=height_rmse.mean(dim=0,keepdim=True)
        slope_mae=slope_mae.mean(dim=0,keepdim=True)
        aspect_mae=aspect_mae.mean(dim=0,keepdim=True)
        slope_rmse=slope_rmse.mean(dim=0,keepdim=True)
        aspect_rmse=aspect_rmse.mean(dim=0,keepdim=True)
    # if None return size B
    return {
        'height_mae': height_mae.cpu().numpy(),
        'height_rmse': height_rmse.cpu().numpy(),
        #'height_max_error': height_max_error.cpu().numpy(),
        'slope_mae':slope_mae.cpu().numpy(),
        'slope_rmse': slope_rmse.cpu().numpy(),
        #'slope_max_error': slope_max_error.cpu().numpy(),
        'aspect_mae':aspect_mae.cpu().numpy(),
        'aspect_rmse':aspect_rmse.cpu().numpy(),
    }

def cal_DEM_metric_single(demA, demB, padding=None, device=None):
    """
    input ndarray or tensor
    :param demA demB: 1 H W
    :param padding: to be cropped
    :reduction: batch mean or none
    :return: {
        'height_mae':
        'slope_mae':
        'aspect_mae':
        'height_rmse':
        'slope_rmse':
        'aspect_rmse':
        }
    """
    assert demA.ndim==3 and demB.ndim==3, "demA and demB should be 3D tensor"

    if padding:
        demA = demA[...,padding:-padding, padding:-padding]
        demB = demB[...,padding:-padding, padding:-padding]
    if isinstance(demA,np.ndarray):
        demA_tensor = torch.from_numpy(demA)
        demB_tensor = torch.from_numpy(demB)
    elif isinstance(demA, torch.Tensor):
        demA_tensor=demA
        demB_tensor=demB

    if device:
        Slope_net.to(device)
        Aspect_net.to(device)
        demA_tensor=demA_tensor.to(device)
        demB_tensor=demB_tensor.to(device)
    else:
        if demA_tensor.is_cuda:
            device=demA_tensor.device
            Slope_net.to(device)
            Aspect_net.to(device)
    with torch.inference_mode():
        demA_slope = Slope_net(demA_tensor)
        demB_slope = Slope_net(demB_tensor)
        demA_aspect = Aspect_net(demA_tensor)
        demB_aspect = Aspect_net(demB_tensor)

    height_mae=torch.abs(demA_tensor - demB_tensor).mean()
    height_rmse=torch.sqrt(torch.mean(torch.pow(demA_tensor - demB_tensor, 2)))

    slope_mae=torch.abs(demA_slope - demB_slope).mean()
    slope_rmse=torch.sqrt(torch.mean(torch.pow(demA_slope - demB_slope, 2)))

    aspect_mae=torch.abs(demA_aspect - demB_aspect).mean()
    aspect_rmse=torch.sqrt(torch.mean(torch.pow(demA_aspect - demB_aspect, 2)))

    return {
        'height_mae': height_mae.cpu().numpy(),
        'height_rmse': height_rmse.cpu().numpy(),
        'slope_mae':slope_mae.cpu().numpy(),
        'slope_rmse': slope_rmse.cpu().numpy(),
        'aspect_mae':aspect_mae.cpu().numpy(),
        'aspect_rmse':aspect_rmse.cpu().numpy(),
    }

def cal_slope_cv(dem,pixel_size=30):
    weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
    weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

    weight1[0][0] = -1
    weight1[0][1] = 0
    weight1[0][2] = 1
    weight1[1][0] = -2
    weight1[1][1] = 0
    weight1[1][2] = 2
    weight1[2][0] = -1
    weight1[2][1] = 0
    weight1[2][2] = 1

    weight2[0][0] = -1
    weight2[0][1] = -2
    weight2[0][2] = -1
    weight2[1][0] = 0
    weight2[1][1] = 0
    weight2[1][2] = 0
    weight2[2][0] = 1
    weight2[2][1] = 2
    weight2[2][2] = 1
    weight1=weight1 / (8 * pixel_size)
    weight2=weight2 / (8 * pixel_size)
    dx=cv2.filter2D(dem, -1, weight1,borderType=cv2.BORDER_CONSTANT)
    dy=cv2.filter2D(dem, -1, weight2,borderType=cv2.BORDER_CONSTANT)
    slope = np.sqrt(np.pow(dx, 2) + np.pow(dy, 2))
    slope = np.arctan(slope) * 180 / math.pi
    return slope


def cal_aspect_cv(dem):
    weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
    weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

    weight1[0][0] = -1
    weight1[0][1] = 0
    weight1[0][2] = 1
    weight1[1][0] = -2
    weight1[1][1] = 0
    weight1[1][2] = 2
    weight1[2][0] = -1
    weight1[2][1] = 0
    weight1[2][2] = 1

    weight2[0][0] = -1
    weight2[0][1] = -2
    weight2[0][2] = -1
    weight2[1][0] = 0
    weight2[1][1] = 0
    weight2[1][2] = 0
    weight2[2][0] = 1
    weight2[2][1] = 2
    weight2[2][2] = 1


    dx=cv2.filter2D(dem, -1, weight1,borderType=cv2.BORDER_CONSTANT)
    dy=cv2.filter2D(dem, -1, weight2,borderType=cv2.BORDER_CONSTANT)

    aspect = math.pi / 180 * np.atan2(-dy, dx)
    # angle from north
    aspect = np.where(aspect > 90, 360 - aspect + 90, 90 - aspect)
    return aspect


def cal_batch_psnr(sr, hr, padding=None, data_range=1.0,reduction="mean"):
    """
    not normailized data
    :param sr: tensor B,1,H,W
    :param hr: tensor B,1,H,W
    :param padding:
    :param data_range: 1.0 for (0,1)
    :return:
    """

    sr, _ = dem_data_convert.tensor4D_maxmin_norm(sr, (0, 1))
    hr, _ = dem_data_convert.tensor4D_maxmin_norm(hr, (0, 1))

    diff = (sr - hr)

    # 裁剪边缘
    if padding:
        diff= diff[..., padding:-padding, padding:-padding]

    res=10. * torch.log10(data_range ** 2 / torch.mean(diff ** 2, dim=(1, 2, 3)))
    if reduction == "mean":
        res = res.mean()
    return res.cpu().numpy()

if __name__=="__main__":
    # test_dem=r'D:\Data\DEM_data\myDEM\ASTGTM2_N42E112_dem.tif'
    #
    # slope=Slope_torch()
    # # device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # # slope.to(device)
    # # aspect=Aspect_torch()
    # # aspect.to(device)
    #
    # input=torch.arange(1, 10).view(1,1,3,3).float()
    # dx,dy= slope.forward_dxdy(input)
    # crop_dx=dx[...,1:-1,1:-1]

    dem_a=torch.randn(2,1,32,32)
    dem_b=torch.randn(2,1,32,32)
    res=cal_DEM_metric(dem_a,dem_b,reduction=None)
    res_mean=cal_DEM_metric(dem_a,dem_b,reduction="mean")
    height_mae=torch.abs(dem_a - dem_b).mean(dim=(1,2,3))


    mean_no=height_mae.numpy()

    mean_yes=torch.mean(height_mae,dim=0,keepdim=True).numpy()

    list1=[]
    list2=[]
    list1.extend(mean_no)
    list2.extend(mean_yes)
    pass