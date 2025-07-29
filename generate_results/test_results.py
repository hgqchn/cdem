import sys
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import copy

import dem_data_convert
import record
import utils
import dem_features

import matplotlib.pyplot as plt

from SIREN import meta_modules, modules

from SIREN.dataset import DEMFolder, ImplicitDownsampled,DEMImplicit_Folder_pair, value_denorm
from mymodel.model import ImplicitModel
from mymodel.SwinIR import SwinEncoder


if __name__ == '__main__':


    utils.seed_everything(utils.default_seed)
    device = utils.default_device

    model_name = "mymodel_swin"
    batchsize = 32

    # 生成结果保存路径
    current_time = utils.get_current_time()
    output_dir=fr'D:\codes\cdem\generate_results\{current_time}'
    utils.make_dir(output_dir)

    # mlp权重参数保存地儿
    mlp_params_save_path=os.path.join(output_dir,'mlp_params')
    utils.make_dir(mlp_params_save_path)

    lrsize = 16
    scale = 4
    hrsize = lrsize * scale

    exp_dir=r'D:\codes\cdem\output\mymodel_swin\2025-07-08_20-55-24'
    ckp_name='best_mymodel_swin_161_2.4091.pth'
    config_file=os.path.join(exp_dir,'config.yaml')
    ckp_file=os.path.join(exp_dir,'checkpoint',ckp_name)

    config=record.get_config_from_yaml(config_file)
    model_config = config['model_config']
    # 模型
    encoder_config = model_config['encoder_config']
    hyper_config = model_config['hyper_config']
    target_config = model_config['target_config']
    target_net = modules.SimpleMLPNetv1(**target_config)
    hyper_net = meta_modules.HyperNetwork(**hyper_config,
                                          target_module=target_net)
    encoder = SwinEncoder(**encoder_config)

    net = ImplicitModel(encoder=encoder,
                        hyper_net=hyper_net,
                        target_net=target_net)
    net.to(device)

    net.load_state_dict(torch.load(ckp_file, map_location=device,weights_only=True))

    # -----------------------------------------#
    # data

    test_dataset = DEMImplicit_Folder_pair(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test',
                                           r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test_NN_120m'
                            )

    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             num_workers=4)

    eval_results = {
        'filename': [],
        'height_mae': [],
        'height_rmse': [],
        'height_rmse_bic': [],
        'height_rmse_error': [],
        'slope_mae': [],
        'slope_rmse': [],
        'aspect_mae': [],
        'aspect_rmse': [],
        # 'miou': []

    }

    # -----------------------------------------#
    # 测试
    net.eval()
    with tqdm(total=len(test_loader), desc=f'test', file=sys.stdout) as t:
        for input, hrdem,hr_coord,hr_value,trans,filename in test_loader:
            b = input.shape[0]
            input = input.to(device)
            hr_coord = hr_coord.to(device)
            hr = hr_value.to(device)
            trans = trans.to(device)

            with torch.inference_mode():
                model_out = net(input, hr_coord)
                sr_value = model_out['model_out']
                mlp_params = model_out['target_params']

            bic_dem= F.interpolate(input, scale_factor=scale, mode='bicubic')
            bic_dem =dem_data_convert.tensor4D_maxmin_denorm(bic_dem,trans)

            sr_value = value_denorm(sr_value, trans)
            hr_dem = value_denorm(hr_value, trans)

            sr_dem = sr_value.view(-1, 1, hrsize, hrsize).detach().cpu()
            hr_dem = hr_dem.view(-1, 1, hrsize, hrsize).detach().cpu()

            eval_res = dem_features.cal_DEM_metric(hr_dem, sr_dem, reduction=None)
            eval_results['filename'].extend(filename)
            for key, value in eval_res.items():
                if value.ndim == 0:
                    eval_results[key].append(value)
                else:
                    eval_results[key].extend(value)

            # bicubic
            eval_res_bic = dem_features.cal_DEM_metric(hr_dem, bic_dem, reduction=None)
            value=eval_res_bic['height_rmse']
            key='height_rmse_bic'
            if value.ndim == 0:
                eval_results[key].append(value)
            else:
                eval_results[key].extend(value)

            rmse_error=np.abs(eval_res['height_rmse'] - eval_res_bic['height_rmse'])
            if rmse_error.ndim == 0:
                eval_results['height_rmse_error'].extend(rmse_error)
            else:
                eval_results['height_rmse_error'].extend(rmse_error)
            # 保存模型参数
            for idx in range(b):
                mlp_param=OrderedDict()
                for name, params in mlp_params.items():
                    mlp_param[name]=mlp_params[name][idx]
                mlp_name=filename[idx]
                target_net.load_state_dict(mlp_param)
                sd=target_net.state_dict()
                save_params={
                    "state_dict": sd,
                    "trans": trans[idx],
                }
                torch.save(save_params, os.path.join(mlp_params_save_path,f'{mlp_name}_trans.pth'))
            t.update(1)


    # 计算平均值
    eval_results['filename'].append('average')
    results_avg = {}
    for key, value in eval_results.items():
        if key =='filename':
            continue
        _mean = np.mean(value)
        eval_results[key].append(_mean)
        results_avg["test/" + key] = _mean
        # eval_results[key].clear()
    results_str = record.compose_kwargs(**results_avg)

    print("Test results:",results_str)

    # 按照height_rmse进行排序，从而可以选择效果最好的dem
    sorted_res=utils.sort_list_dict(eval_results, 'height_rmse')
    record.save_dict_csv_pd(sorted_res,save_file=os.path.join(output_dir,'test_results.csv'))



    # 不能直接保存参数，会很大
    # target_net.load_state_dict(mlp_param)
    # utils.get_parameter_nums(target_net)
    # sd=target_net.state_dict()
    # torch.save(sd, os.path.join(mlp_params_save_path, f'test.pth'))
    #
    # mlp_param_test = mlp_param
    # for name,value in mlp_param_test.items():
    # mlp_param_test[name]=value.detach()
    # torch.save(mlp_param_test, os.path.join(mlp_params_save_path, f'test1.pth'))
    #
    # sd_test=torch.load(os.path.join(mlp_params_save_path, f'test1.pth'))
    #
    # total=0
    # for name, param in sd_test.items():
    #     print(name, param.shape)
    #     total += param.numel()*param.element_size()
    # print(total)
    #
    # total=0
    # for name, param in mlp_param_test.items():
    #     print(name, param.shape)
    #     total += param.numel()*param.element_size()
    # print(total)

    # sd=torch.load(os.path.join(mlp_params_save_path, f'dem2_1026_mlp.pth'))
    # target_net.load_state_dict(sd)
    # hr_coord=hr_coord[0]
    # out= target_net(hr_coord)

    # mlp_parms=torch.load(os.path.join(mlp_params_save_path, f'dem2_1026_trans.pth'),weights_only=False)
    # sd=mlp_parms['state_dict']
    # trans=mlp_parms['trans']
    # target_net.load_state_dict(sd)
    # hr_coord=hr_coord[0]
    # model_out= target_net(hr_coord,return_grad=True)
    # out=model_out['model_out']
    # dy_dx= model_out['dy_dx']
    # out=out.unsqueeze(0)  # Add batch dimension
    # trans= trans.unsqueeze(0)
    # hr_denorm=value_denorm(out,trans)
    # hr_dem= hr_denorm.view(1,hrsize,hrsize).detach().cpu().numpy()
    pass
