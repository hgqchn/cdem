import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from torch.utils.data import DataLoader

import logging
import wandb
from tqdm import tqdm

import copy

import record
import utils
import dem_features

from SIREN import meta_modules
from SIREN.dataset import DEMFolder, ImplicitDownsampled,value_denorm

#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    debug=True
    if debug:
        outfile=False
        wandb_mode='disabled'
    else:
        outfile=True
        wandb_mode='online'

    recorder=record.Recorder(logger=logging.getLogger(__name__),
                             output_path=r'D:\codes\cdem\output\SIREN',
                             use_tensorboard=False,
                             outfile=outfile,
                             )
    logger=recorder.get_logger()

    if outfile:
        #模型权重文件保存路径
        save_path=recorder.save_path
        model_save_path=os.path.join(save_path,'checkpoint')
        utils.make_dir(model_save_path)
    epoch_save = 100

    utils.seed_everything(utils.default_seed)
    device=utils.default_device

    model_name="SIREN"
    lr=1e-4
    epochs=500
    batchsize=64

    current_time = utils.get_current_time()


    wandb_save_path = recorder.save_path
    os.environ['WANDB_DIR'] = wandb_save_path

    wandb.init(
        project="CDEM",
        name=f"{model_name}_{current_time}",
        notes="SIREN demo",
        tags=[f"{model_name}", "Local"],
        mode=wandb_mode, #online disabled offline
        config={
            "model": model_name,
            "lr": lr,
            "epochs": epochs,
            "batchsize": batchsize,
        }
    )
    lrsize=16
    scale=4
    hrsize=lrsize*scale


    config={
        "model_name":model_name,
        "lr":lr,
        "epochs":epochs,
        "batchsize":batchsize,
        "scale": scale,
        "hrsize": hrsize,
    }


    embed_dim=512
    model_config={
        "in_features": 1,
        "out_features": 1,
        "image_resolution": (16, 16),
        "embed_dim": embed_dim,

        "target_hidden": 32,
        "target_hidden_layers": 2,
        "use_pe": False,
        "num_frequencies": 8,
        "use_hsine": True,


        "hyper_hidden_layers": 3,
        "hyper_hidden_features": embed_dim,
    }

    config["model_config"]=model_config
    wandb.config.update({"model_config": model_config})
    if recorder.outfile:
        recorder.save_config_to_yaml(config)

    #-----------------------------------------#
    # data
    train_folder=DEMFolder(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Train')
    train_dataset=ImplicitDownsampled(
        dataset=train_folder,
        scale=4,
    )


    train_loader = DataLoader(train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=4)


    test_folder=DEMFolder(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test',
                          )
    test_dataset=ImplicitDownsampled(
        dataset=test_folder,
        scale=4,
    )
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              num_workers=4)
    #-----------------------------------------#
    # 模型，优化器，调度器


    model=meta_modules.ConvolutionalNeuralProcessImplicit2DHypernet(
        **model_config
    )
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

    start_epoch=1
    # for train
    epoch_loss=utils.AverageMeter()
    # for test
    # epoch_rmse=utils.AverageMeter()
    # epoch_mae=utils.AverageMeter()
    eval_results = {
        'height_mae': [],
        'height_rmse': [],
        'slope_mae': [],
        'slope_rmse': [],
        'aspect_mae': [],
        'aspect_rmse': [],
        #'miou': []
    }
    best_epoch=0
    best_rmse=float('inf')
    best_model = copy.deepcopy(model.state_dict())


    #-----------------------------------------#
    for epoch in range(start_epoch,epochs+1):
        model.train()
        # 训练
        epoch_loss.reset()
        with tqdm(total=len(train_loader), desc=f'epoch {epoch}/{epochs} train', file=sys.stdout) as t:
            for input,hr_coord,gt,trans in train_loader:
                b=input.shape[0]
                input=input.to(device)
                hr_coord=hr_coord.to(device)
                gt=gt.to(device)
                trans=trans.to(device)

                #assert utils.check_model_nan(model.encoder) == False, "Encoder has NaN values"

                model_out= model(input,hr_coord)
                sr_value=model_out['model_out']

                loss=criterion(sr_value,gt)
                optimizer.zero_grad()

                loss.backward()
                if model_config['use_hsine']:
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss.update(loss.item(),b)
                t.set_postfix_str(f"loss: {epoch_loss.avg:.6f}")
                t.update(1)


        logger.info(f'Epoch {epoch}/{epochs}, '
                    f"loss: {epoch_loss.avg:.6f}" )

        wandb.log({
            "train/loss": epoch_loss.avg,
        }, step=epoch)


        lr_scheduler.step()

        # 测试
        model.eval()
        # epoch_mae.reset()
        # epoch_rmse.reset()

        with tqdm(total=len(test_loader), desc=f'epoch {epoch}/{epochs} test', file=sys.stdout) as t:
            for input,hr_coord,gt,trans in test_loader:
                b=input.shape[0]
                input=input.to(device)
                hr_coord=hr_coord.to(device)
                gt=gt.to(device)
                trans=trans.to(device)


                with torch.inference_mode():
                    model_out = model(input, hr_coord)
                    sr_value = model_out['model_out']

                sr_value=value_denorm(sr_value,trans)
                gt=value_denorm(gt,trans)

                sr_dem=sr_value.view(-1,1,hrsize,hrsize).detach().cpu()
                gt=gt.view(-1,1,hrsize,hrsize).detach().cpu()
                eval_res = dem_features.cal_DEM_metric(gt, sr_dem)
                for key, value in eval_res.items():
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
        logger.info(f'test results: {results_str}')

        wandb.log(
            results_avg, step=epoch
        )

        this_rmse_avg = results_avg['test/height_rmse']
        if this_rmse_avg < best_rmse:
            best_rmse = this_rmse_avg
            best_epoch = epoch
            best_model = copy.deepcopy(model.state_dict())

        if outfile:
            if epoch % epoch_save == 0 or epoch == epochs:
                model_path = os.path.join(model_save_path, f'{model_name}_{epoch}.pth')
                torch.save(model.state_dict(), model_path)
    if outfile:
        best_model_path = os.path.join(model_save_path, f'best_{model_name}_{best_epoch}_{best_rmse:.4f}.pth')
        torch.save(best_model, best_model_path)
        logger.info(f'best model at {best_epoch} saved: {best_model_path}')

    wandb.finish()

    pass
