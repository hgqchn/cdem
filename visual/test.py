import sys
import os

from utils import tools,other_imresize
from dem_utils import dem_data_io
import matplotlib.pyplot as plt

if __name__ == '__main__':

    test_dem=r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_2\dem2_2831.TIF'
    dem=dem_data_io.read_dem(test_dem)

    hr_dem=dem
    lr_dem=other_imresize.cv2_imresize(hr_dem,1/4,mode="nearest")


    fig,axes=plt.subplots(1,2)
    cax=axes[0].imshow(hr_dem,cmap="terrain")
    axes[0].set_title("HR")
    axes[1].imshow(lr_dem,cmap="terrain")
    axes[1].set_title("LR")
    for ax in axes.flatten():
        ax.axis('off')
    #plt.colorbar(cax,ax=axes)
    plt.show(block=True)
