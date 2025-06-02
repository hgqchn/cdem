import sys
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.utils.data import default_collate as collate_fn
from EBCF_CDEM.harmonic_embedding import HarmonicEmbedding
from EBCF_CDEM.edsr import EDSR
from utils import get_pixel_center_coord_tensor


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]  #记录除最后维度以外的维度
        x = self.layers(x.view(-1, x.shape[-1]))  #x.shape[-1]应该与in_dim相同
        return x.view(*shape, -1) #恢复与输入相同的维度

class EBCF(nn.Module):
    def __init__(
        self,
        make_coord_local=True,
        interp_mode='nearest', #or 'none'
    ):
        '''
        :param encoder:编码器网络 edsr.EDSR
        :param biasnet:偏差网络 MLP 多层感知机 每层神经元数量256 256 256 256
        :param data_sub:
        :param data_div:
        :param make_coord_local: true
        :param interp_mode: 偏差预测
        :param posEmbeder: 位置编码器
        '''
        super().__init__()

        self.make_coord_local = make_coord_local
        self.interp_mode = interp_mode


        self.encoder=EDSR(n_resblocks=16, n_feats=64, res_scale=1,
               input_dim=1)
        # 输入2 输出2*(2*16+1)=66
        self.posEmbeder = HarmonicEmbedding(16)
        # 66+64=130
        mlp_in_dim= self.posEmbeder.get_output_dim(2) + self.encoder.out_dim
        self.mlp=MLP(mlp_in_dim,1, [256, 256, 256, 256]) #偏差网络

    #根据偏差网络查询偏差值
    def query_bias(self, q_samples):
        #q_samples: b,q,dim
        bs, q = q_samples.shape[:2]
        # mlp输入形状 b*q,dim 输出形状 b,q,1
        # 恢复成b,q,1
        pred = self.mlp(q_samples.view(bs * q, -1)).view(bs, q, -1)
        return pred


    #编码器生成特征
    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    #生成隐藏向量
    def gen_latvec(self, sample_grid, mode='bicubic'):
        feat = self.feat #b,C,Hin,Win

        grid_ = sample_grid.clone() #b,H*W,2
        #将输入的feat特征映射到grid_网格上
        #
        # grid_sample函数 输入 b,c,Hin,Win, grid:b,H,W,2 输出b c H W
        #下文中输入的feat大小为b,C,Hin,Win
        # grid_.flip(-1).unsqueeze(1)形状为b,1,HW,2
        # 输出形状为 b,C,1,HW
        gen_latvec = F.grid_sample(
            #flip(-1) 最后一个维度翻转，将H，W坐标值翻转，因为原先是(纵轴，横轴)，转为（横轴，纵轴）
            # unsqueeze(1) 增加一个维度b,1,H*W,2
            feat, grid_.flip(-1).unsqueeze(1),
            mode=mode, align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        #输出形状为 b,C,1,HW -> b,C,HW -> b,HW,C
        # C就是encoder的特征维度 64

        return gen_latvec

    # 预测计算
    def pred(self, inp, local_coord):
        # 根据输入，由编码器生成特征
        # 注意：输入是整个低分辨率图像

        # local_coord是对应于高分辨率图像的坐标
        # b H_hr*W_hr 2
        self.gen_feat(inp)

        # 感觉这里有问题
        if self.make_coord_local:
            # 生成隐藏向量
            latvec = self.gen_latvec(
                sample_grid=local_coord,
            )
            # b H_hr*W_hr 2
            coord_ = local_coord.clone()
            # feat大小b,c,H,W
            # 先生成feat形状的网格坐标，且不展平 make_coord返回大小为H W 2
            # 范围默认在-1,1
            # 然后调换顺序 2 H W
            # 然后增加维度1 2 H W
            # expand扩展，扩展为b,2,H,W
            feat_coord = get_pixel_center_coord_tensor(self.feat.shape[-2:], flatten=False).to(inp.device) \
                .permute(2, 0, 1) \
                .unsqueeze(0).expand(self.feat.shape[0], 2, *self.feat.shape[-2:])
            # 对低分图像大小的网格进行采样，得到对应于低分图像的查询坐标q_coord
            # 输入input：feat_coord b 2 H_lr W_lr
            # grid：b 1 H_hr*W_hr 2 输出形状为1 H_hr*W_hr,表示有这么多查询点，也就高分图像的像素点数量
            # coord_数组翻转，coord_: b H_hr*W_hr,2 最后一个维度的数值对应为先H后W。
            # 翻转最后一个维度,因为grid_sample的参数grid最后一维是input的坐标（x,y）先横轴坐标，也就是列索引。而翻转前是先行索引，后列索引
            # unsqueeze(1) 在第1个维度前插入一个维度b 1 H_hr*W_hr 2 符合输入形状
            # align_corners=False (-1,-1)表示input的左上像素的左上点
            # grid_sample返回b 2 1 H_hr*W_hr
            # 压缩掉第三维度，b 2 H_hr*W_hr
            # 调换顺序b H_hr*W_hr 2 最终的查询坐标
            q_coord = F.grid_sample(
                feat_coord, coord_.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            # 好像是是固定值
            # 相对坐标？感觉不对
            # Liif的做法
            # 形状为b,q,2
            rel_coord = local_coord - q_coord
            rel_coord[:, :, 0] *= self.feat.shape[-2]  # H
            rel_coord[:, :, 1] *= self.feat.shape[-1]  # W
            # =带位置嵌入
            if self.posEmbeder:
                # b,q,2 输出b,q,pos_dim
                # pos_dim 66
                rel_coord = self.posEmbeder(rel_coord)
            # b,q,dim
            q_samples = torch.cat([latvec, rel_coord], dim=-1)
        else:
            latvec = self.gen_latvec(
                sample_grid=local_coord,
            )
            rel_coord = local_coord
            if self.posEmbeder:
                rel_coord = self.posEmbeder(rel_coord)
            q_samples = torch.cat([latvec, rel_coord], dim=-1)

        # q_应该是查询的意思
        bias_value = self.query_bias(q_samples)


        # get horiz line for preding sdf
        grid_ = local_coord.clone()
        # 输入低分图像插值到高分，获取查询点处的高程值作为基准值
        horiz_line = F.grid_sample(
            inp, grid_.flip(-1).unsqueeze(1),
            mode=self.interp_mode, align_corners=False
        )[:, :, 0, :].permute(0, 2, 1)
        # 插值得到的基准值加上预测得到的偏差值
        elev_value = horiz_line + bias_value

        return elev_value,bias_value,horiz_line

    # 前向计算
    def forward(self, lr,coord):
        #   lr 归一化后的值
        #   shape:B,1,H,W
        #   coord: B,L,2  L=H*W
        device=lr.device

        # 预测值
        preds = self.pred(
            inp=lr,
            local_coord=coord  #B,L,2
        )

        return preds

    def super_resolution(self,lr,scale):
        # lr: 归一化到0,1
        #   shape:B,1,H,W
        B,_,h,w=lr.shape
        device=lr.device
        data_sub = self.data_sub.to(device)
        data_div = self.data_div.to(device)

        inp = (lr - data_sub) / data_div
        H,W = int(h*scale), int(w*scale)
        # 高分网格的坐标 H*W,2
        hr_coord= get_pixel_center_coord_tensor([H, W], flatten=True).to(device)
        # B,H*W,2
        hr_coord=hr_coord.unsqueeze(0).expand(B,-1,-1)

        res=self.pred(
            inp=inp,
            local_coord=hr_coord  # B,H*W,2
        )
        # B,H*W,1
        hr_value,_,_=res
        # 将 hr_value 从 [B, H*W, 1] 重塑为 [B, 1, H, W]
        hr = hr_value.reshape(B, H, W, 1).permute(0, 3, 1, 2)

        return hr

if __name__ == '__main__':
    pass
