import sys
import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader, Dataset
from SIREN import modules
import matplotlib.pyplot as plt

import utils,dem_data_convert

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))



class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input 允许计算关于输入的导数
        output = self.net(coords)
        return output, coords

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




# 一次性读取图像所有像素点的坐标和值
class DEMFitting(Dataset):
    def __init__(self, dem):
        super().__init__()
        assert dem.ndim == 2
        assert dem.shape[-1] == dem.shape[-2]

        # C,H,W
        dem_t = torch.from_numpy(dem).unsqueeze(0)
        data_norm, trans = dem_data_convert.tensor_maxmin_norm(dem_t)
        self.trans=trans.unsqueeze(0)

        self.pixels = data_norm.permute(1, 2, 0).view(-1, 1)
        self.coords = utils.get_pixel_center_coord_tensor(dem.shape[0])

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels

def value_denorm(norm_value, trans):

    # value: B,L,1
    # trans: B,3

    # 确保输入张量在同一设备上
    if norm_value.device != trans.device:
        trans = trans.to(norm_value.device)

    # B 3->B 1-> B 1 1
    data_min= trans[:, 0].reshape(-1, 1, 1)
    norm_min= trans[:, 1].reshape(-1, 1, 1)
    scale= trans[:, 2].reshape(-1, 1, 1)
    # B L 1
    new_data= (norm_value - norm_min) / scale + data_min
    return new_data

if __name__ == '__main__':
    import utils
    hr_file_path = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_0\dem0_0.TIF'
    hr_dem = utils.read_dem(hr_file_path)

    dataset=DEMFitting(hr_dem)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True, num_workers=0)

    model=modules.SimpleMLPNet(hidden_features=64,num_hidden_layers=3,image_resolution=(16,16),use_pe=True,num_frequencies=16)

    model.to('cuda')

    # total_steps=20
    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    save_path=r'./out_simpleMLP_PE_16'
    os.makedirs(save_path, exist_ok=True)

    #plt.imsave(os.path.join(save_path, 'dem_0.png'),arr=hr_dem,cmap='terrain')

    optim = torch.optim.Adam(lr=1e-4, params=model.parameters())
    # 1,65536,2  1,65536,1
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()


    for step in range(total_steps):
        # 一次性把所有像素值输入
        model_out= model(model_input)
        model_output=model_out['model_out']
        coords=model_out['model_in']

        #model_output, coords = model_out

        loss = ((model_output - ground_truth) ** 2).mean()

        if not step % steps_til_summary or step==total_steps-1:
            print(f"Step {step}, Total loss {loss:6f}" )
            img_grad = gradient(model_output, coords)
            img_dx=img_grad[...,1]
            img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            out_denorm=value_denorm(model_output,dataset.trans)

            axes[0].imshow(out_denorm.cpu().view(64, 64).detach().numpy(),cmap="terrain")
            # 梯度的模
            axes[1].imshow(img_grad.norm(dim=-1).cpu().view(64, 64).detach().numpy())
            #axes[2].imshow(img_laplacian.cpu().view(64, 64).detach().numpy())
            axes[2].imshow(img_dx.cpu().view(64, 64).detach().numpy())

            plt.savefig(os.path.join(save_path, f"{step}.png"))
            plt.close(fig)
        optim.zero_grad()
        loss.backward()
        optim.step()
    torch.save(model.state_dict(), os.path.join(save_path, 'siren_dem0_0.pth'))

    pass
