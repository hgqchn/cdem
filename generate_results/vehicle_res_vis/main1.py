import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import utils


def read_txt(file_path):
    """
    Reads a text file and returns its content as a list of dictionaries.
    Each line in the file should be formatted as 'key: value, key: value, ...'.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            fields = line.strip().split(",")
            entry = {}
            for field in fields:
                key, value = field.split(":")
                entry[key.strip()] = float(value.strip())
            data.append(entry)

    dtype = [('time', float), ('vel', float), ('posx', float),
             ('posy', float), ('posz', float), ('ticktime(ms)', float)]

    # 结构化数组
    structured_data = np.array([tuple(d.values()) for d in data], dtype=dtype)

    return structured_data




if __name__ == '__main__':

    dem_name=r'dem0_0.TIF'
    dem_idx=0
    dem_file=fr'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_{dem_idx}\{dem_name}'
    dem = utils.read_dem(dem_file)


    hr_file=r'files1\\6dof_output_hr.txt'
    hr_data = read_txt(hr_file)[:1500]
    hr_x= hr_data['posx']
    hr_y = hr_data['posy']
    hr_z=hr_data['posz']

    hr_vel=hr_data['vel']
    hr_zero_indices = np.where(hr_vel == 0)[0]

    plt.figure()
    plt.imshow(dem, cmap='terrain')
    plt.scatter(hr_x/30+5, hr_y/30+5,s=1,color='red',label='hr',marker='*')  # s=点的大小
    plt.axis('off')
    #plt.title('Trajectory with fixed direction')
    plt.savefig('固定路径.png', bbox_inches='tight', pad_inches=0)
    plt.show()
