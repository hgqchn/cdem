import numpy as np
import torch

def img_nparray_2D_2tensor3D(img):
    """
    numpy array shape: H,W
    numpy array dtype: float32
    :param img: input numpy array
    :return: torch tensor C,H,W  float32
    """
    # H W 1
    img=np.expand_dims(img,-1)
    # 1 H W
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

def nparray2tensor(data):
    # numpy array to torch tensor
    return torch.from_numpy(np.ascontiguousarray(data)).float()

def data_maxmin_norm(data, norm_range=(-1,1), epsilon=1e-6,minmax_height=None):
    """
    DEM data max-min normalization.
    :param data: numpy array
    :param epsilon:
    :param norm_range: (min,max) of normalized data
    :return:  trans: np.array([data_min,norm_min,scale])
            (norm_data-norm_min)/scale+data_min
    """
    norm_min=norm_range[0]
    norm_max=norm_range[1]
    if minmax_height:
        data_min=minmax_height[0]
        data_max=minmax_height[1]
    else:
        data_max=data.max()
        data_min=data.min()
    norm_data= (norm_max-norm_min) * (data - data_min) /(data_max - data_min + epsilon) + norm_min
    scale=(norm_max-norm_min)/(data_max - data_min + epsilon)
    trans=np.array([data_min,norm_min,scale]).astype(np.float32)
    return norm_data, trans


def data_maxmin_denorm(norm_data,trans):
    """
    :param norm_data: numpy array
    :param trans: np.array([data_min,norm_min,scale])
    :return: normalized numpy array
    """
    data_min=trans[0]
    norm_min=trans[1]
    scale=trans[2]
    data= (norm_data - norm_min) / scale + data_min
    return data

def array_maxmin_tensor(data, norm_range=(-1, 1), epsilon=1e-6):
    """

    :param data: H W dem numpy array
    :param norm_range:
    :param epsilon:
    :return: normlized tensor size (1 H W), trans tensor size (3)
    """
    # H W dem numpy data -> 1 H W tensor

    norm_data, data_trans = data_maxmin_norm(data,norm_range=norm_range, epsilon=epsilon)
    # H W ndarray-> 1 H W tensor
    input_tensor = img_nparray_2D_2tensor3D(norm_data)
    #
    data_trans_tensor = nparray2tensor(data_trans)
    # input tensor: 1 H W
    # data_trans_tensor: 3
    return input_tensor, data_trans_tensor

def tensor_maxmin_norm(data, norm_range=(-1, 1), epsilon=1e-6,minmax_height=None):
    """
    DEM data max-min normalization.
    for single DEM
    :param data: tensor： 1 H W or 1 1 H W
    :param epsilon:
    :param norm_range: (min,max) of normalized data
    :param minmax_height: given min max of data
    :return:  trans: [data_min,norm_min,scale])
            (norm_data-norm_min)/scale+data_min
    """
    assert data.ndim == 3 or (data.ndim == 4 and data.shape[0]==1)
    norm_min=norm_range[0]
    norm_max=norm_range[1]
    if minmax_height:
        data_min=minmax_height[0]
        data_max=minmax_height[1]
    else:
        data_max=data.max()
        data_min=data.min()
    norm_data= (norm_max-norm_min) * (data - data_min) /(data_max - data_min + epsilon) + norm_min
    scale=(norm_max-norm_min)/(data_max - data_min + epsilon)
    trans=torch.tensor([data_min,norm_min,scale])
    # 1 1 H W
    if len(data.shape)==4:
        # 1,3
        trans=trans.unsqueeze(0)
    return norm_data, trans

def tensor4D_maxmin_norm(data, norm_range=(-1, 1), epsilon=1e-6,minmax_height=None):
    """
    DEM data max-min normalization.
    based on each sample's max and min or
    :param data: tensor：B 1 H W
    :param epsilon:
    :param norm_range: (min,max) of normalized data
    :return:
        norm_data, B 1 H W
        trans: B 3
            [data_min,norm_min,scale])
            ### (norm_data-norm_min)/scale+data_min
    """
    norm_min=norm_range[0]
    norm_max=norm_range[1]
    batchsize=data.shape[0]
    if minmax_height:
        data_min = torch.full([batchsize], minmax_height[0], dtype=data.dtype)
        data_max = torch.full([batchsize], minmax_height[1], dtype=data.dtype)
    else:
        # B
        data_max = data.amax(dim=(1, 2, 3))
        # B
        data_min = data.amin(dim=(1, 2, 3))

    # 将标量扩展为与数据相同形状进行广播计算
    data_min_expanded = data_min.view(batchsize, 1, 1, 1)
    data_max_expanded = data_max.view(batchsize, 1, 1, 1)

    # B 1 H W
    norm_data = (norm_max - norm_min) * (data - data_min_expanded) / (
            data_max_expanded - data_min_expanded + epsilon) + norm_min

    # B 3
    trans = torch.stack([data_min,
                         torch.full_like(data_min, norm_min),
                         (norm_max - norm_min) / (data_max - data_min + epsilon)], dim=1)

    return norm_data, trans


def tensor_maxmin_trans(data,trans):
    """
    for single DEM norm use trans
    :param data: tensor： 1 H W
    :param trans: tensor： 3
    :return: tensor： 1 H W
    """
    data_min=trans[0]
    norm_min=trans[1]
    scale=trans[2]
    norm_data=(data-data_min)*scale+norm_min
    return norm_data

def tensor4D_maxmin_denorm(norm_data, trans):
    """
    dem tensor data maxmin denormalization

    :param norm_data: B 1 H W
    :param trans: B 3
    :return: B 1 H W tensor
    """
    # 确保输入张量在同一设备上
    if norm_data.device != trans.device:
        trans = trans.to(norm_data.device)

    # B 3->B 1-> B 1 1 1
    data_min= trans[:, 0].reshape(-1, 1, 1, 1)
    norm_min= trans[:, 1].reshape(-1, 1, 1, 1)
    scale= trans[:, 2].reshape(-1, 1, 1, 1)
    # B 1 H W
    new_data= (norm_data - norm_min) / scale + data_min
    return new_data


