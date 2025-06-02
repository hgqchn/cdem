import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SIREN import meta_modules,dataset,modules
from utils import to_pixel_samples_tensor,torch_imresize
import utils
import dem_data_convert,dem_features

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


# 其实就是边缘检测罢了，sobel算子
class Slope_map(nn.Module):
    def __init__(self):

        super(Slope_map, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

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

        # nn.Parameter 注册为模型参数
        self.weight1 =nn.Parameter(torch.tensor(weight1)) # 自定义的权值
        self.weight2 =nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        self.bias.requires_grad = False
    def forward(self, x):
        # x 为归一化的输入
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))

        return slope



if __name__ == '__main__':

    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 提前初始化 CUDA 上下文
    if torch.cuda.is_available():
        torch.zeros(1).cuda()  # 会触发 CUDA 上下文创建

    weight_path=r"D:\codes\cdem\output\SIREN\test\2025-05-29_16-23-22\mlp_params\dem0_0_params.pth"
    model=modules.SimpleMLPNet(hidden_features=64,num_hidden_layers=3,image_resolution=(16,16))
    sd=torch.load(weight_path,weights_only=True)
    for name,params in sd.items():
        sd[name].squeeze_(0)
    torch.save(sd,weight_path)
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    height=64
    width=64
    scale=4
    hr_coord = utils.get_pixel_center_coord_tensor((height, width))
    hr_coord = hr_coord.to(device).unsqueeze(0)  # Add batch dimension

    out=model(hr_coord)
    coord_grad=out['model_in']
    model_out=out['model_out']



    lr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m\dem_0\dem0_0.TIF'
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    hr_dem = utils.read_dem(hr_file_path)
    hr_dem = torch.from_numpy(hr_dem).unsqueeze(0).unsqueeze(0)
    lr_dem = utils.read_dem(lr_file_path)
    lr_dem = torch.from_numpy(lr_dem).unsqueeze(0).unsqueeze(0)
    bic_dem = F.interpolate(lr_dem, scale_factor=scale, mode='bicubic')

    slope_map = Slope_map()
    hr_slope=slope_map(hr_dem)
    bic_slope=slope_map(bic_dem)

    _, trans = dem_data_convert.tensor_maxmin_norm(lr_dem, (-1, 1), 1e-6,
                                                       None)
    sr_value = dataset.value_denorm(model_out, trans)
    slope= gradient(sr_value, coord_grad, grad_outputs=torch.ones_like(model_out))

    slope_value=torch.norm(slope, dim=-1)
    # 输入坐标也是归一化的
    # 归一化大-1,1
    # 需要除以 128=64*2

    sr_dem = sr_value.view(1, 1, height, width)
    slope_dem = slope_value.view(1, 1, height, width)
    # eval_res = dem_features.cal_DEM_metric(sr_dem, hr_dem, device=device)


#     import numpy as np
#     save_path=r'D:\codes\cdem\output\SIREN\test\test_dem'
#     save_file_path=os.path.join(save_path,"trans.csv")
#     trans_np=trans.cpu().numpy()
#     np.savetxt(save_file_path,trans_np,delimiter=',',header='min_value, norm_min, scale',comments='')
# #   bic_dem=bic_dem.squeeze(0).squeeze(0).numpy()
#
#     import imageio
#     imageio.imwrite(os.path.join(save_path, "bic_dem.tif"), bic_dem, format='TIFF')





    pass
