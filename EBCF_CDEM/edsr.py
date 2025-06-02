# Enhanced deep residual networks for single image super-resolution.
# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import torch.nn as nn


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias) #保证输入输出不变,默认使用偏置


class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, n_resblocks=16, n_feats=64, res_scale=1,
             input_dim=1):
        super(EDSR, self).__init__()

        n_resblocks = n_resblocks
        n_feats = n_feats
        self.out_dim=n_feats
        kernel_size = 3

        act = nn.ReLU(True)


        # define head module
        m_head = [default_conv(input_dim, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                default_conv, n_feats, kernel_size, act=act, res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(default_conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)


    def forward(self, x):

        x = self.head(x)

        res = self.body(x)
        x=res+x

        return x






