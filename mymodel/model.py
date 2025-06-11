import sys
import os
import torch
from torch import nn
from collections import OrderedDict

from mymodel import modules,hyper_model


class ImplicitModel(nn.Module):
    def __init__(self, encoder,hyper_net,target_net):
        super().__init__()
        self.encoder = encoder
        self.hyper_net = hyper_net
        self.target_net = target_net

    # 修改
    def forward(self, input_img,coords):

        embedding = self.encoder(input_img)
        target_params = self.hyper_net(embedding)
        model_output = self.target_net(coords, params=target_params)

        return {'coord': model_output['model_in'],
                'model_out': model_output['model_out'],
                'latent_vec': embedding,
                'target_params': target_params}

    def get_target_net_weights(self, input_img):
        embedding = self.encoder(input_img)
        target_params = self.hyper_net(embedding)
        return target_params, embedding

    def save_target_net_weights(self,input_img,filename_list):
        b=input_img.size(0)
        embedding = self.encoder(input_img)
        target_params = self.hyper_net(embedding)
        for i in range(b):
            filename = filename_list[i]
            params = OrderedDict()
            for name, param in target_params.items():
                params[name] = param[i].unsqueeze(0)
            torch.save(params, filename)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False

if __name__ == '__main__':
    pass
