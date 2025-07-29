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

    dtype = [('slope',float),('time', float), ('vel', float), ('posx', float),
             ('posy', float), ('posz', float), ('ticktime(ms)', float)]

    # 结构化数组
    structured_data = np.array([tuple(d.values()) for d in data], dtype=dtype)

    return structured_data




if __name__ == '__main__':

    dem_name=r'dem0_0.TIF'
    dem_idx=0
    dem_file=fr'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test\dem_{dem_idx}\{dem_name}'
    dem = utils.read_dem(dem_file)


    hr_file=r'files\\6dof_output_hr.txt'
    hr_data = read_txt(hr_file)[:1640]
    hr_x= hr_data['posx']
    hr_y = hr_data['posy']
    hr_z=hr_data['posz']
    hr_slope=hr_data['slope']
    hr_vel=hr_data['vel']
    hr_zero_indices = np.where(hr_vel == 0)[0]

    plt.figure()
    plt.imshow(dem, cmap='terrain')
    plt.scatter(hr_x/30+20, hr_y/30+10,s=0.1,color='red',label='hr',marker='o')  # s=点的大小
    plt.axis('off')
    #plt.title('Trajectory with path planning')
    plt.savefig('路径规划.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    # lr_file=r'files\\6dof_output_lr.txt'
    # lr_data = read_txt(lr_file)[:1640]
    # lr_x= lr_data['posx']
    # lr_y = lr_data['posy']
    # lr_z=lr_data['posz']
    # lr_slope=lr_data['slope']
    # lr_vel=lr_data['vel']
    # lr_zero_indices = np.where(lr_vel == 0)[0]
    #
    #
    # sr_file=r'files\\6dof_output_sr.txt'
    # sr_data = read_txt(sr_file)[:1640]
    # sr_x= sr_data['posx']
    # sr_y = sr_data['posy']
    # sr_z=sr_data['posz']
    # sr_slope=sr_data['slope']
    # sr_vel=sr_data['vel']
    # sr_zero_indices = np.where(sr_vel == 0)[0]
    #
    # mlp_file=r'files\\6dof_output_mlp.txt'
    # mlp_data = read_txt(mlp_file)[:1640]
    # mlp_x= mlp_data['posx']
    # mlp_y = mlp_data['posy']
    # mlp_z=mlp_data['posz']
    # mlp_slope=mlp_data['slope']
    # mlp_vel=mlp_data['vel']
    # mlp_zero_indices = np.where(mlp_vel == 0)[0]
    #
    # lr_x_mae= np.mean(np.abs(lr_x - hr_x))
    # lr_y_mae = np.mean(np.abs(lr_y - hr_y))
    # lr_z_mae = np.mean(np.abs(lr_z - hr_z))
    # lr_slope_mae = np.mean(np.abs(lr_slope - hr_slope))
    # print(f"lr_x_mae: {lr_x_mae:.6f},\n"
    #       f"lr_y_mae: {lr_y_mae:.6f},\n"
    #       f"lr_z_mae: {lr_z_mae:.6f},\n"
    #       f"lr_slope_mae: {lr_slope_mae:.6f}")
    #
    # sr_x_mae= np.mean(np.abs(sr_x - hr_x))
    # sr_y_mae = np.mean(np.abs(sr_y - hr_y))
    # sr_z_mae = np.mean(np.abs(sr_z - hr_z))
    # sr_slope_mae = np.mean(np.abs(sr_slope - hr_slope))
    # print(f"sr_x_mae: {sr_x_mae:.6f},\n"
    #       f"sr_y_mae: {sr_y_mae:.6f},\n"
    #       f"sr_z_mae: {sr_z_mae:.6f},\n"
    #       f"sr_slope_mae: {sr_slope_mae:.6f}")
    # mlp_x_mae= np.mean(np.abs(mlp_x - hr_x))
    # mlp_y_mae = np.mean(np.abs(mlp_y - hr_y))
    # mlp_z_mae = np.mean(np.abs(mlp_z - hr_z))
    # mlp_slope_mae = np.mean(np.abs(mlp_slope - hr_slope))
    # print(f"mlp_x_mae: {mlp_x_mae:.6f},\n"
    #       f"mlp_y_mae: {mlp_y_mae:.6f},\n"
    #       f"mlp_z_mae: {mlp_z_mae:.6f},\n"
    #       f"mlp_slope_mae: {mlp_slope_mae:.6f}")



    pass
