import sys
import os
import torch
import torch.nn as nn


class CosineActivation(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class target_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.act=CosineActivation()
        self.fc1=nn.Linear(in_features=2, out_features=32,bias=True)
        self.fc2=nn.Linear(in_features=32,out_features=64,bias=True)
        self.fc3=nn.Linear(in_features=64,out_features=256,bias=True)
        self.fc4=nn.Linear(in_features=256,out_features=64,bias=True)
        self.fc5=nn.Linear(in_features=64,out_features=1,bias=True)

    def forward(self, x):
        x=self.act(self.fc1(x))
        x=self.act(self.fc2(x))
        x=self.act(self.fc3(x))
        x=self.act(self.fc4(x))
        x=self.fc5(x)
        return x

class hyper_net(nn.Module):
    def __init__(self,in_channel=1,image_size=64):
        super().__init__()
        self.in_channel=in_channel
        self.image_size=image_size
        self.relu=nn.ReLU()
        self.inception_module=nn.ModuleList(
            [
                nn.Conv2d(in_channel,in_channel,1),
                nn.Conv2d(in_channel,in_channel,3,1,1),
                nn.Conv2d(in_channel,in_channel,5,1,2),
                nn.AvgPool2d(3,1,1),
            ]
        )

        in_channel=in_channel*4
        self.common=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.BatchNorm2d(in_channel),
        )
        self.common_conv=nn.Sequential(
            nn.Conv2d(in_channel,2*in_channel,3,1,1),
            nn.ReLU(),
            nn.BatchNorm2d(2*in_channel),
            nn.MaxPool2d(2),
        )

        in_channel=2*in_channel
        image_size=image_size//2
        self.fc1_weights=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel * image_size * image_size, 2 * 32),

        )
        self.fc1_bias=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel * image_size * image_size, 32),

        )
        self.fc2_weights=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel * image_size * image_size, 32*64),

        )
        self.fc2_bias=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel * image_size * image_size, 64),

        )
        self.fc3_weight=nn.Sequential(
            nn.Conv2d(in_channel,2*in_channel,3,1,1),
            nn.ReLU(),
            nn.Conv2d(2*in_channel,4*in_channel,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*in_channel * image_size * image_size, 64*256),

        )
        self.fc3_bias=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel * image_size * image_size, 256),

        )
        self.fc4_weight=nn.Sequential(
            nn.Conv2d(in_channel,2*in_channel,3,1,1),
            nn.ReLU(),
            nn.Conv2d(2*in_channel,4*in_channel,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4*in_channel * image_size * image_size, 64*256),

        )
        self.fc4_bias=nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel * image_size * image_size, 64),

        )

        self.fc5_weight=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel * image_size * image_size, 64),

        )
        self.fc5_bias=nn.Sequential(
            nn.Conv2d(in_channel,in_channel,3,1,1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_channel*image_size*image_size,1),

        )



    def forward(self,x):
        first_out=[]
        for module in self.inception_module:
            first_out.append(self.relu(module(x)))
        out1=torch.cat(first_out,1)
        out2=self.common(out1)
        out=out1+out2
        out=self.common_conv(self.relu(out))

        return {
            'fc1_weights':self.fc1_weights(out),
            'fc1_bias':self.fc1_bias(out),
            'fc2_weights':self.fc2_weights(out),
            'fc2_bias':self.fc2_bias(out),
            'fc3_weight':self.fc3_weight(out),
            'fc3_bias':self.fc3_bias(out),
            'fc4_weight':self.fc4_weight(out),
            'fc4_bias':self.fc4_bias(out),
            'fc5_weight':self.fc5_weight(out),
            'fc5_bias':self.fc5_bias(out)
        }


def apply_hyper_params(target_model, hyper_params):
    """
    将超网络生成的参数应用到目标网络上

    参数:
        target_model: 目标网络实例
        hyper_params: 超网络生成的参数字典
    """
    # 重塑参数并应用到目标网络

    # FC1 层
    target_model.fc1.weight.data = hyper_params['fc1_weights'].reshape(32,2)
    target_model.fc1.bias.data = hyper_params['fc1_bias']

    # FC2 层
    target_model.fc2.weight.data = hyper_params['fc2_weights'].reshape(64,32)
    target_model.fc2.bias.data = hyper_params['fc2_bias']

    # FC3 层
    target_model.fc3.weight.data = hyper_params['fc3_weight'].reshape(256,64)
    target_model.fc3.bias.data = hyper_params['fc3_bias']

    # FC4 层
    target_model.fc4.weight.data = hyper_params['fc4_weight'].reshape(64,256)
    target_model.fc4.bias.data = hyper_params['fc4_bias']

    # FC5 层
    target_model.fc5.weight.data = hyper_params['fc5_weight'].reshape(1,64)
    target_model.fc5.bias.data = hyper_params['fc5_bias']


def process_batch(hyper_out, input_batch, coords):
    """

    :param hyper_out:
    :param input_batch:
    :param coords: B,L,2
    :return:
    """
    batch_size = input_batch.shape[0]
    results = []

    # 获取超网络对整个batch的输出
    hyper_params_batch = hyper_out

    device=input_batch.device
    for i in range(batch_size):
        # 为每个样本创建一个新的目标网络
        model_instance = target_net().to(device)

        # 提取当前样本的参数
        sample_params = {
            k: v[i] for k, v in hyper_params_batch.items()
        }

        # 应用参数到目标网络
        apply_hyper_params(model_instance, sample_params)

        # 对应每个样本的 L,2
        coord=coords[i].squeeze()
        # 使用配置好的网络进行前向计算
        result = model_instance(coord)
        result=result.unsqueeze(0)
        results.append(result)

    # 将所有结果拼接成一个batch
    return torch.cat(results, dim=0)

if __name__ == '__main__':
    # import torch, torch.nn as nn
    #
    # x = torch.randn(32, 2)  # (B=32, in_features=128)
    # net=target_net()
    # y = net(x)
    # print(y.shape)  # ✅ (32, 1)

    hyper_net=hyper_net(1,64)
    input_exp=torch.randn(2,1,64,64)
    out=hyper_net(input_exp)

    from utils import get_pixel_center_coord_tensor
    coord=get_pixel_center_coord_tensor((128, 128), flatten=True)
    coord=coord.unsqueeze(0).expand(8,-1,-1)

    res=process_batch(out,input_exp,coord)

    pass
