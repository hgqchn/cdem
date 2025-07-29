import sys
import os

import utils
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 单个DEM
# HR,LR,Bic
if __name__ == '__main__':

    # test_dem=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_2\dem2_1026.TIF'
    # test_dem=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_2\dem2_969.TIF'
    # test_dem=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_1\dem1_1110.TIF'
    test_dem = r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_6\dem6_799.TIF'
    dem=utils.read_dem(test_dem)

    hr_dem=dem
    lr_dem=utils.cv2_imresize(hr_dem,1/4,mode="nearest")
    bic_dem=utils.torch_imresize(lr_dem,4,mode="bicubic")

    all_max = max(np.max(hr_dem), np.max(lr_dem), np.max(bic_dem))
    all_min = min(np.min(hr_dem), np.min(lr_dem), np.min(bic_dem))

    vmin = 5 * np.floor(all_min / 5)
    vmax = 5 * np.ceil(all_max / 5)

    ims = []
    fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
    ims.append(axes[0].imshow(hr_dem,vmin=vmin,vmax=vmax,cmap="terrain"))
    axes[0].set_title("HR")
    ims.append(axes[1].imshow(lr_dem,vmin=vmin,vmax=vmax,cmap="terrain"))
    axes[1].set_title("LR")
    ims.append(axes[2].imshow(bic_dem,vmin=vmin,vmax=vmax,cmap="terrain"))
    axes[2].set_title("Bicubic")

    for ax in axes.flatten():
        ax.axis('off')



    # 添加一个 colorbar，挂在最后一个im对象上，也可以直接挂在第一个
    cbar = fig.colorbar(ims[-1], ax=axes, orientation='vertical',shrink=0.7)
    cbar.set_label('Colorbar Label')

    plt.savefig('test.svg')
    #plt.colorbar(cax,ax=axes)
    plt.show(block=True)
