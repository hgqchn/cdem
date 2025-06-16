import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time

from test_fit.fit_siren import Siren, laplace, gradient, divergence
from SIREN.modules import PosEncodingNeRF2D
import utils


# code from SIREN


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    returns: shape (sidelen**2,2)
        (y,x) coordinates in a range of -1 to 1.
    '''

    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)
    return img

def reverse_tensor(img_tensor):
    """
    将图像张量的像素值从[-1, 1]转换到[0, 1]
    :param img_tensor: 输入图像张量，形状为(C, H, W)
    :return: 转换后的图像张量，形状为(C, H, W)
    """
    img_tensor=img_tensor*0.5+0.5
    img_tensor=img_tensor*255.0
    return img_tensor

# 一次性读取图像所有像素点的坐标和值
class ImageFitting(Dataset):
    def __init__(self, sidelength):
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels


class SIRENwithPE(nn.Module):
    def __init__(self,
                 in_features=2,
                 out_features=1,
                 num_frequencies=6,
                 hidden_features=256,
                 hidden_layers=3,
                 outermost_linear=True):
        super(SIRENwithPE, self).__init__()
        self.pe = PosEncodingNeRF2D(num_frequencies)
        pe_output_dim = self.pe.get_output_dim()
        self.siren = Siren(in_features=pe_output_dim, out_features=out_features,
                           hidden_features=hidden_features, hidden_layers=hidden_layers,
                           outermost_linear=outermost_linear)

    def forward(self, x):
        x = self.pe(x)
        return self.siren(x)




if __name__ == '__main__':



    output_dir = r'D:\codes\cdem\test_fit_out\image_fit_out'

    original_img_tensor = get_cameraman_tensor(64)
    original_img_tensor = original_img_tensor*0.5+0.5  # 将像素值从[-1, 1]转换到[0, 1]
    original_img_tensor = original_img_tensor * 255.0  # 将像素值从[0, 1]转换到[0, 255]
    # original_img = original_img_tensor.permute(1, 2, 0).numpy()  # 转换为HWC格式
    # original_img = original_img.astype(np.uint8)  # 转换为无符号8位整数类型
    # img = Image.fromarray(original_img.squeeze(-1))  # 去掉通道维度
    # img.save(os.path.join(output_dir,'original_64.png'))

    current_time = utils.get_current_time()
    img_size=64
    output_dir = os.path.join(output_dir, current_time + f"img_{img_size}_SIREN")
    utils.make_dir(output_dir)
    cameraman = ImageFitting(img_size)
    # 数据集的大小就是1，所以只有一个batch
    dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True, num_workers=0)

    img_siren = Siren(in_features=2, out_features=1, hidden_features=256,
                      hidden_layers=3, outermost_linear=True)
    #img_siren = SIRENwithPE()
    # img_siren.cuda()
    img_siren.to('cuda')

    total_steps = 500  # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = 10

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())
    # 1,65536,2  1,65536,1
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    for step in range(1, total_steps + 1):
        # 一次性把所有像素值输入
        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth) ** 2).mean()

        model_output= reverse_tensor(model_output)
        if step % steps_til_summary == 0:
            print("Step %d, Total loss %f" % (step, loss))
            img_grad = gradient(model_output, coords)
            img_laplacian = laplace(model_output, coords)

            fig, axes = plt.subplots(1, 6, figsize=(30, 6))
            for ax in axes:
                ax.axis('off')
            axes[0].imshow(original_img_tensor.cpu().view(img_size, img_size).detach().numpy())
            axes[0].set_title("original image")
            axes[1].imshow(model_output.cpu().view(img_size, img_size).detach().numpy())
            axes[1].set_title(f"fit image {loss}")
            # 梯度的模
            axes[2].imshow(img_grad.norm(dim=-1).cpu().view(img_size, img_size).detach().numpy())
            axes[2].set_title("gradient norm")
            axes[3].imshow(img_grad[..., 1].cpu().view(img_size, img_size).detach().numpy())
            axes[3].set_title("gradient dx")
            axes[4].imshow(img_grad[..., 0].cpu().view(img_size, img_size).detach().numpy())
            axes[4].set_title("gradient dy")

            axes[5].imshow(img_laplacian.cpu().view(img_size, img_size).detach().numpy())
            axes[5].set_title("laplacian")

            plt.savefig(os.path.join(output_dir, f"{step}.png"))
            plt.close()

        optim.zero_grad()
        loss.backward()
        optim.step()

    pass
