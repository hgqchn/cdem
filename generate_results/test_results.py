import sys
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import copy
import record
import utils
import dem_features

from SIREN import meta_modules, modules

from SIREN.dataset import DEMFolder, ImplicitDownsampled, value_denorm
from mymodel.model import ImplicitModel
from mymodel.SwinIR import SwinEncoder




if __name__ == '__main__':


    utils.seed_everything(utils.default_seed)
    device = utils.default_device

    model_name = "mymodel_swin"
    lr = 1e-4
    epochs = 300
    batchsize = 32

    current_time = utils.get_current_time()



    lrsize = 16
    scale = 4
    hrsize = lrsize * scale

    config = {
        "model_name": model_name,
        "lr": lr,
        "epochs": epochs,
        "batchsize": batchsize,
        "scale": scale,
        "hrsize": hrsize,
        "content_loss": 1.0,
        "slope_loss": 0.5,
        "fft_loss": 0.2,
    }

    laten_code_dim = 256
    swin_encoder_config = {
        "img_size": 16,
        "patch_size": 1,
        "in_chans": 1,
        "embed_dim": laten_code_dim,
        "depths": [4, 4, 4, 4],
        "num_heads": [4, 4, 4, 4],
        "window_size": 1,
        "mlp_ratio": 4.0,
    }

    target_config = {
        "out_features": 1,
        "hidden_features": 32,
        "num_hidden_layers": 2,
        "use_pe": False,
        "num_frequencies": 10,
        "use_hsine": True,
    }

    hyper_config = {
        "hyper_in_features": laten_code_dim,
        "hyper_hidden_layers": 3,
        "hyper_hidden_features": laten_code_dim,
    }

    model_config = {}
    model_config["encoder_config"] = swin_encoder_config
    model_config["target_config"] = target_config
    model_config["hyper_config"] = hyper_config
    config["model_config"] = model_config


    # -----------------------------------------#
    # data

    test_folder = DEMFolder(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test',
                            )
    test_dataset = ImplicitDownsampled(
        dataset=test_folder,
        scale=4,
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=32,
                             shuffle=False,
                             pin_memory=True,
                             drop_last=False,
                             num_workers=4)
    # -----------------------------------------#
    # 模型，优化器，调度器

    target_net = modules.SimpleMLPNetv1(**target_config)
    hyper_net = meta_modules.HyperNetwork(**hyper_config,
                                          target_module=target_net)
    encoder = SwinEncoder(**swin_encoder_config)

    net = ImplicitModel(encoder=encoder,
                        hyper_net=hyper_net,
                        target_net=target_net)
    net.to(device)


    eval_results = {
        'height_mae': [],
        'height_rmse': [],
        'slope_mae': [],
        'slope_rmse': [],
        'aspect_mae': [],
        'aspect_rmse': [],
        # 'miou': []
    }



    # -----------------------------------------#
    # 测试
    net.eval()
    # epoch_mae.reset()
    # epoch_rmse.reset()

    with tqdm(total=len(test_loader), desc=f'test', file=sys.stdout) as t:
        for input, hr_coord, gt, trans in test_loader:
            b = input.shape[0]
            input = input.to(device)
            hr_coord = hr_coord.to(device)
            gt = gt.to(device)
            trans = trans.to(device)

            with torch.inference_mode():
                model_out = net(input, hr_coord)
                sr_value = model_out['model_out']

            sr_value = value_denorm(sr_value, trans)
            gt = value_denorm(gt, trans)

            sr_dem = sr_value.view(-1, 1, hrsize, hrsize).detach().cpu()
            gt = gt.view(-1, 1, hrsize, hrsize).detach().cpu()
            eval_res = dem_features.cal_DEM_metric(gt, sr_dem, reduction=None)
            for key, value in eval_res.items():
                if isinstance(value, list):
                    eval_results[key].extend(value)
                else:
                    eval_results[key].append(value)

            # height_mae = torch.abs(sr_value - gt).mean(dim=(1, 2)).mean()
            # height_rmse = torch.sqrt(torch.mean(torch.pow(sr_value - gt, 2), dim=(1, 2))).mean()
            #
            # epoch_mae.update(height_mae.item(),b)
            # epoch_rmse.update(height_rmse.item(),b)
            # t.set_postfix_str(f"rmse: {epoch_rmse.avg:.6f}  "
            #                   f"mae: {epoch_mae.avg:.6f}")
            t.update(1)

    results_avg = {}
    for key, value in eval_results.items():
        _mean = np.mean(value)
        results_avg["test/" + key] = _mean
        eval_results[key].clear()
    results_str = record.compose_kwargs(**results_avg)

    this_rmse_avg = results_avg['test/height_rmse']




    pass
