
import sys
import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Dataset
from SIREN import modules
import matplotlib
import matplotlib.pyplot as plt

import utils,dem_data_convert

#matplotlib.use('Agg')
#plt.ioff()

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

if __name__ == '__main__':

    # model=modules.SimpleMLPNet(hidden_features=64,num_hidden_layers=3,image_resolution=(16,16))
    # weight_path = r"D:\codes\cdem\test_fit_data\out\siren_dem0_0.pth"
    # sd= torch.load(weight_path, weights_only=True)
    # model.load_state_dict(sd)
    # coords = utils.get_pixel_center_coord_tensor(64).unsqueeze(0)
    #
    # model_out = model(coords)
    # model_output = model_out['model_out']
    # coords_out = model_out['model_in']
    # img_grad = gradient(model_output, coords_out)
    # img_grad_dx=img_grad[..., 1]
    # img_grad_dy=img_grad[..., 0]
    # img_grad_norm=img_grad.norm(dim=-1).cpu().view(64, 64).detach().numpy()
    # #plt.imshow(img_grad_dx.view(64, 64).detach().numpy())
    #
    # fig, axes = plt.subplots(1, 1, figsize=(6, 3))
    # axes.imshow(img_grad_norm)
    # #axes.axis('off')
    # plt.savefig(f"test.png")
    # plt.close(fig)


    for i in range(10):
        fig, axes = plt.subplots(1, 1, figsize=(6, 3))
        axes.imshow(np.random.rand(64, 64))




    pass
