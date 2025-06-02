import sys
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


import logging
import wandb
from tqdm import tqdm

import copy

from EBCF_CDEM import dataset,ebcf
import record
import utils
import dem_features


def value_denorm(norm_value, trans):

    # value: B,L,1
    # trans: B,3

    # 确保输入张量在同一设备上
    if norm_value.device != trans.device:
        trans = trans.to(norm_value.device)

    # B 3->B 1-> B 1 1
    data_min= trans[:, 0].reshape(-1, 1, 1)
    norm_min= trans[:, 1].reshape(-1, 1, 1)
    scale= trans[:, 2].reshape(-1, 1, 1, 1)
    # B L 1
    new_data= (norm_value - norm_min) / scale + data_min
    return new_data

if __name__ == '__main__':
    recorder=record.Recorder(logger=logging.getLogger(__name__),
                             output_path=r'D:\codes\cdem\output\EBCF_CDEM',
                             use_tensorboard=False,
                             )
    logger=recorder.get_logger()

    #模型权重文件保存路径
    save_path=recorder.save_path
    model_save_path=os.path.join(save_path,'checkpoint')
    utils.make_dir(model_save_path)
    epoch_save = 10

    utils.seed_everything(utils.default_seed)
    device=utils.default_device

    model_name="EBCF-CDEM"
    lr=1e-4
    epochs=300
    batchsize=32

    current_time = utils.get_current_time()


    wandb_save_path = recorder.save_path
    os.environ['WANDB_DIR'] = wandb_save_path

    wandb.init(
        project="CDEM",
        name=f"{model_name}_{current_time}",
        notes="移植官方代码",
        tags=[f"{model_name}", "Local"],
        mode="online", #disabled offline
        config={
            "model": model_name,
            "lr": lr,
            "epochs": epochs,
            "batchsize": batchsize,
        }
    )

    config={
        "model_name":model_name,
        "lr":lr,
        "epochs":epochs,
        "batchsize":batchsize,
    }
    recorder.save_config_to_yaml(config)

    train_folder=dataset.DEMFolder(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Train',
                             repeat=4)
    train_dataset=dataset.SDFImplicitDownsampled(
        dataset=train_folder,
        scale_max=4,
        image_size=16,
        sample_q=256,
    )

    test=train_dataset[0]
    train_loader = DataLoader(train_dataset,
                              batch_size=batchsize,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              num_workers=4)


    test_folder=dataset.DEMFolder(r'D:\Data\DEM_data\dataset_TfaSR\(60mor120m)to30m\DEM_Test',
                             repeat=1)
    test_dataset=dataset.SDFImplicitDownsampled(
        dataset=test_folder,
        scale_min=4,
        scale_max=4,
        image_size=None,
        sample_q=None,
    )
    test_loader = DataLoader(test_dataset,
                            batch_size=1,
                              shuffle=False,
                              pin_memory=True,
                              drop_last=False,
                              num_workers=4)
    #-----------------------------------------#
    # 模型，优化器，调度器
    model=ebcf.EBCF()
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200], gamma=0.1)

    start_epoch=1
    # for train
    epoch_loss=utils.AverageMeter()
    # for test
    epoch_rmse=utils.AverageMeter()
    epoch_mae=utils.AverageMeter()
    #-----------------------------------------#
    for epoch in range(start_epoch,epochs):
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

                sr_value,_,_ = model(input,hr_coord)

                loss=criterion(sr_value,gt)
                optimizer.zero_grad()
                loss.backward()
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
        epoch_mae.reset()
        epoch_rmse.reset()

        with tqdm(total=len(test_loader), desc=f'epoch {epoch}/{epochs} test', file=sys.stdout) as t:
            for input,hr_coord,gt,trans in test_loader:
                b=input.shape[0]
                input=input.to(device)
                hr_coord=hr_coord.to(device)
                gt=gt.to(device)
                trans=trans.to(device)

                with torch.inference_mode():
                    sr_value,_,_ = model(input,hr_coord)

                sr_value=value_denorm(sr_value,trans)
                gt=value_denorm(gt,trans)
                height_mae = torch.abs(sr_value - gt).mean(dim=(1, 2)).mean()
                height_rmse = torch.sqrt(torch.mean(torch.pow(sr_value - gt, 2), dim=(1, 2))).mean()

                epoch_mae.update(height_mae.item(),b)
                epoch_rmse.update(height_rmse.item(),b)
                t.set_postfix_str(f"rmse: {epoch_rmse.avg:.6f}"
                                  f"mae: {epoch_mae.avg:.6f}")
                t.update(1)



        logger.info(f'Epoch {epoch}/{epochs}, '
                    f"rmse: {epoch_rmse.avg:.6f}  "
                    f"mae: {epoch_mae.avg:.6f}")


        wandb.log(
            {
                "test/rmse": epoch_rmse.avg,
                "test/mae": epoch_mae.avg,
            }, step=epoch
        )

        if epoch % epoch_save == 0 or epoch == epochs:
            model_path = os.path.join(model_save_path, f'ebcf-cdem_{epoch}.pth')
            torch.save(model.state_dict(), model_path)

    wandb.finish()

    pass
