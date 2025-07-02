import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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

    structured_data = np.array([tuple(d.values()) for d in data], dtype=dtype)

    return structured_data



if __name__ == '__main__':

    hr_file=r'6dof_output_hr.txt'
    hr_data = read_txt(hr_file)
    hr_x= hr_data['posx'][:200][::10]  # 只取前100个点
    hr_y = hr_data['posy'][:200][::10]  # 只取前100个点

    plt.scatter(hr_x, hr_y,s=1,color='red',label='hr',marker='*')  # s=点的大小
    
    lr_file=r'6dof_output_lr.txt'
    lr_data = read_txt(lr_file)
    lr_x= lr_data['posx'][:200][::10]
    lr_y = lr_data['posy'][:200][::10]

    plt.scatter(lr_x, lr_y,s=1,color='blue',label='lr',marker='^')

    mlp_file=r'6dof_output_sr.txt'
    mlp_data = read_txt(mlp_file)
    mlp_x= mlp_data['posx'][:200][::10]
    mlp_y = mlp_data['posy'][:200][::10]

    plt.scatter(mlp_x, mlp_y,s=1,color='green',label='mlp',marker='x')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Multi-Trajectory Scatter')
    #plt.grid(True)
    #plt.axis('equal')
    plt.show()




    pass
