import sys
import os
import numpy as np
import torch
import torch.nn as nn

class Slope_torch_forloss(nn.Module):
    def __init__(self,eps=1e-6):
        #eps在计算损失函数时设置，防止梯度计算时出现nan
        super(Slope_torch_forloss, self).__init__()
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
        weight1 = weight1 / 8
        weight2 = weight2 / 8
        # nn.Parameter 注册为模型参数
        self.weight1 =nn.Parameter(torch.tensor(weight1)) # 自定义的权值
        self.weight2 =nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))  # 自定义的偏置
        self.weight1.requires_grad = False
        self.weight2.requires_grad = False
        self.bias.requires_grad = False
        self.eps=eps

    def forward(self, x):
        dx = torch.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = torch.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        # 这里的坡度没有任何单位
        slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2)+self.eps)

        return slope

class SlopeLoss(nn.Module):
    def __init__(self):
        super(SlopeLoss, self).__init__()
        self.feature=Slope_torch_forloss()
        self.criterion = nn.MSELoss()
    def forward(self, input, gt):
        slope=self.feature(input)
        gt_slope=self.feature(gt)

        loss = self.criterion(slope,gt_slope)
        return loss

def fft_mse_loss(img1, img2):
    img1_fft = torch.fft.fftn(img1, dim=(2,3),norm="ortho")
    img2_fft = torch.fft.fftn(img2, dim=(2,3),norm="ortho")
    # Splitting x and y into real and imaginary parts
    x_real, x_imag = torch.real(img1_fft), torch.imag(img1_fft)
    y_real, y_imag = torch.real(img2_fft), torch.imag(img2_fft)
    # Calculate the MSE between the real and imaginary parts separately
    mse_real = torch.nn.MSELoss()(x_real, y_real)
    mse_imag = torch.nn.MSELoss()(x_imag, y_imag)
    return mse_imag+mse_real


if __name__ == '__main__':
    pass
