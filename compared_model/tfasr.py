import torch
import torch.nn as nn
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN
#from mmcv.ops import ModulatedDeformConv2dPack as DCN


class residualBlock(nn.Module):
    def __init__(self, in_channels=64, k=3, n=64, s=1):
        super(residualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n, k, stride=s, padding=1)
        self.bn1 = nn.BatchNorm2d(n)
        self.relu=nn.ReLU()
        self.conv2 = nn.Conv2d(n, n, k, stride=s, padding=1)
        self.bn2 = nn.BatchNorm2d(n)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        return self.bn2(self.conv2(y)) + x


class Net(nn.Module):
    def __init__(self, n_residual_blocks, upsample_factor):
        super(Net, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.upsample_factor = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        for i in range(self.n_residual_blocks):
            self.add_module('residual_block' + str(i + 1), residualBlock())

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 1, 3, 1, 1)
        self.conv4 = nn.Conv2d(1, 1, 1, stride=1, padding=0)

    def forward(self, x):
        #########################original version########################
        x = self.conv1(x)

        y = x.clone()
        for i in range(self.n_residual_blocks):
            y = self.__getattr__('residual_block' + str(i + 1))(y)

        x = self.bn2(self.conv2(y)) + x

        x = self.relu(self.dconv2_2(x))
        x = self.relu(self.dconv2_3(x))
        return self.conv4(self.dconv2_4(x))

class Netv2(nn.Module):
    def __init__(self, n_residual_blocks=16, upsample_factor=4):
        super(Netv2, self).__init__()
        self.n_residual_blocks = n_residual_blocks
        self.scale = upsample_factor

        self.conv1 = nn.Conv2d(1, 64, 9, stride=1, padding=4)

        residual_blocks=[]
        for i in range(self.n_residual_blocks):
            residual_blocks.append(residualBlock())
        self.residual = nn.Sequential(*residual_blocks)

        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dconv2_2 = DCN(64, 64, 3, 1, 1)
        self.dconv2_3 = DCN(64, 64, 3, 1, 1)
        self.dconv2_4 = DCN(64, 64, 3, 1, 1)
        # 上采样块
        self.upscale = nn.Sequential(
            nn.Conv2d(64, 64 * (upsample_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upsample_factor)
        )
        self.conv4 = nn.Conv2d(64, 1, 1, stride=1, padding=0)

    def forward(self, x):

        x = self.conv1(x)

        y = x.clone()
        y=self.residual(y)

        x = self.bn2(self.conv2(y)) + x

        x = self.relu(self.dconv2_2(x))
        x = self.relu(self.dconv2_3(x))
        x = self.relu(self.dconv2_4(x))
        x = self.upscale(x)
        return self.conv4(x)


if __name__ == '__main__':
    from utils import record
    logger=record.custom_logger(__name__)
    model = Net(16, 4)
    model_pth = r'D:\codes\My_SR\run_model_dem\TfaSR\2024-11-13_21-03-02\tfasr_model.pth'
    model.load_state_dict(torch.load(model_pth, weights_only=True))
    #print(model)
