import sys
import os
import torch
from sympy import denom
from torch.utils.data import DataLoader

import logging
from tqdm import tqdm
import numpy as np

from utils import tools,settings,record
from dem_utils import dem_data_convert,dem_features,dem_class_tools
from models.diffusion_utils import space_timesteps,make_beta_schedule

import wandb


def get_dataset(dataset_path_str):
    if dataset_path_str is None:
        raise Exception('dataset_path_str cannot be None')
    return tools.get_object_from_path_v2(dataset_path_str)

class BaseRunner():
    def __init__(self,config):
        self.device = settings.default_device
        self.start_time = tools.get_current_time()

        if isinstance(config, str):
            if not config.endswith(".yaml"):
                raise Exception('config file is not a yaml file')
            config=record.get_config_from_yaml(config)
        self.config = config
        self.output_path=tools.get_path_from_root(config.get("output_path","output/default"))

        self.model_save_epoch=50

        self.model_name=config["model_name"]
        self.scale=config["scale"]
        self.lr=config.get("lr",None)
        self.epochs=config.get("epochs",None)
        self.timesteps=config.get("timesteps",None)
        self.batchsize=config.get("batchsize",None)

        # for diffusion sr model
        self.unet_config=config.get("unet_config",None)
        self.pretrained_config=config.get("pretrained_config",None)
        self.diffusion_config=config.get("diffusion_config",None)

        # for other sr model
        self.sr_model_config=config.get("model_config",None)

        # for eval river miou
        self.river_miou = dem_class_tools.RiverMIoU().to(self.device)



        self.flag=config.get("flag","train")

        self.train_loader_config = config.get("train_loader_config", None)
        self.test_loader_config = config.get("test_loader_config", None)

        self.train_dataset=None
        self.test_dataset=None
        self.train_loader=None
        self.test_loader=None
        # 在单独的函数中初始化，不能置空，后续利用hasattr判断
        # self.recorder=None
        # self.logger=None
        # self.wandb_run=None


        self.init_all()

    def init_seeds(self):
        settings.seed_everything(seed=settings.default_seed)

    def init_all(self):

        self.init_seeds()
        if self.flag=="train":
            self.init_train_loader()
            self.init_test_loader()
        elif self.flag=="test":
            self.init_test_loader()

        if self.pretrained_config is not None:
            self.init_pretrained_sr()
        if self.unet_config is not None:
            self.init_denoise_network()
        if self.diffusion_config is not None:
            self.init_diffusion()
        if self.flag=="train":
            self.set_optimizer_denoise_nerwork()

        if self.sr_model_config is not None:
            self.model=self.init_model(self.sr_model_config)

    def init_train_loader(self):
        new_kwargs=self.train_loader_config.copy()
        self.train_dataset = get_dataset(self.train_loader_config["dataset"])
        new_kwargs["dataset"] =self.train_dataset
        self.train_loader = DataLoader(**new_kwargs)

    def init_test_loader(self):
        new_kwargs=self.test_loader_config.copy()
        self.test_dataset = get_dataset(self.test_loader_config["dataset"])
        new_kwargs["dataset"] = self.test_dataset
        self.test_loader = DataLoader(**new_kwargs)

    def init_recorder(
            self,
            logger=logging.getLogger(__name__),
            output_path=None,
            save_with_time=True,
            extra_name=None,
            outfile=True,
            use_tensorboard=False):
        if output_path is None:
            output_path = self.output_path
        recorder=record.Recorder(
            logger=logger,
            output_path=output_path,
            save_with_time=save_with_time,
            extra_name=extra_name,
            outfile=outfile,
            use_tensorboard=use_tensorboard
        )
        self.recorder=recorder
        self.logger=recorder.get_logger()
        self.writer=recorder.get_writer()
        if self.recorder.outfile:
            self.recorder.save_config_to_yaml(self.config)




    def init_wandb(self,wandb_config=None,project="DEM Super-Resolution",name=None,notes="",mode="online",tags=["Local"]):
        # mode: online offline disabled
        if wandb_config is None:
            wandb_config={
                "scale": self.scale,
                "epochs": self.epochs,
                "lr": self.lr,
                "timesteps": self.timesteps,
                "batchsize": self.batchsize,
            }
        if name is None:
            name=f"{self.model_name}_{self.start_time}"

        if hasattr(self, "recorder"):
            wandb_save_path = self.recorder.save_path
        else:
            wandb_save_path = os.path.join(self.output_path, f'{self.start_time}')

        os.environ['WANDB_DIR']=wandb_save_path
        wandb_run=wandb.init(
            project=project,
            name=name,
            notes=notes,
            mode=mode,
            tags=[self.model_name]+tags,
            config=wandb_config,
        )
        self.wandb_run=wandb_run
        if self.pretrained_config is not None and self.pretrained_config is not None and self.diffusion_config is not None:
            self.wandb_run.config.update({"pretrain_config":self.pretrained_config})
            self.wandb_run.config.update({"unet_config":self.unet_config})
            self.wandb_run.config.update({"diffusion_config":self.diffusion_config})


    def init_pretrained_sr(self):
        cls=tools.get_object_from_path_v2(self.pretrained_config["path"])
        sr_net=cls(**self.pretrained_config["model_kwargs"])

        try:
            if self.pretrained_config["weight_path"] is not None:
                state_dict=torch.load(self.pretrained_config["weight_path"],weights_only=True)
                sr_net.load_state_dict(state_dict)
            else:
                raise ValueError("weight_path is None")
        except Exception as e:
            print(f"pretrained model not load state dict: {e}")
        self.pretrained_sr_net=sr_net


    def init_denoise_network(self):
        class_path=self.unet_config["path"]
        kwargs=self.unet_config["model_kwargs"]
        denoise_network=tools.instantiate_object(class_path,**kwargs)
        self.denoise_network=denoise_network

    def load_state_dict(self, weight_path):
        self.denoise_network.load_state_dict(torch.load(weight_path, weights_only=True))

    def init_diffusion(self):
        class_path=self.diffusion_config["path"]
        cls=tools.get_object_from_path_v2(class_path)

        betas_config=self.diffusion_config["betas_config"]
        betas=make_beta_schedule(n_timestep=self.timesteps,**betas_config)

        other_kwargs=self.diffusion_config["other_kwargs"]

        diffusion_model=cls(
            denoise_network=self.denoise_network,
            betas=betas,
            initial_up_network=self.pretrained_sr_net,
            **other_kwargs
        ).to(self.device)
        self.diffusion_model=diffusion_model

    def set_optimizer_denoise_nerwork(self):
        if not hasattr(self, "denoise_network"):
            raise ValueError("Denoise network not initialized")
        self.optimizer=torch.optim.Adam(
            self.denoise_network.parameters(),
            lr=self.lr)

        # lr_scheduler

    def init_model(self,model_config):
        class_path=model_config["model_path"]
        kwargs=model_config.get("model_kwargs",None)
        if kwargs is None:
            kwargs={}
        model=tools.instantiate_object(class_path,**kwargs)
        model.to(self.device)
        return model

    def set_optimizer(self,model):
        self.optimizer=torch.optim.Adam(
            model.parameters(),
            lr=self.lr
        )
        # lr_sceluder

    def train(self):
        for epoch in range(1,self.epochs+1):

            self.diffusion_model.train()

            log,cost_time=self.train_step(epoch=epoch)
            log_str = ""
            for key, value in log.items():
                log_str = log_str + f"{key}={value:.6f}  "
            log_str+= f"[{cost_time}]"
            if hasattr(self, "lr_scheduler"):
                self.lr_scheduler.step()
            if hasattr(self, "logger"):
                self.logger.info(f"epoch:{epoch}/{self.epochs},training: "+log_str)
            else:
                print(f"epoch:{epoch}/{self.epochs},training: "+log_str)
            if hasattr(self, "wandb_run"):
                self.wandb_run.log(log,step=epoch)

            if epoch % self.model_save_epoch == 0 or epoch == self.epochs:
                self.save_model(epoch, log["train_loss"])


    def train_step(self,epoch):
        device=self.device
        epoch_loss = tools.AverageMeter()
        with tqdm(total=len(self.train_loader), desc=f'Epoch {epoch}/{self.epochs} training', file=sys.stdout) as pbar:
            for data,filename in self.train_loader:
                lr, hr, trans = data
                lr, hr = lr.to(device), hr.to(device)

                self.optimizer.zero_grad()
                loss = self.diffusion_model(lr, hr)
                loss.backward()
                self.optimizer.step()
                epoch_loss.update(loss.detach().cpu().numpy())
                pbar.set_postfix(loss='{:.6f}'.format(epoch_loss.avg))
                pbar.update(1)
            cost_time=tools.second_to_min_sec(pbar.format_dict['elapsed'])
        log={
            'train_loss': epoch_loss.avg,
        }
        return log,cost_time



    def test_step(self,epoch=0):
        if epoch==0:
            epoch=self.epochs

        eval_results = {
            'height_mae': [],
            'height_rmse': [],
            'slope_mae': [],
            'slope_rmse': [],
            'aspect_mae': [],
            'aspect_rmse': [],
            'miou': []
        }
        self.diffusion_model.eval()

        test_loader=self.test_loader
        device=self.device
        with tqdm(total=len(test_loader), desc='Test', file=sys.stdout) as t:
            for data, name in test_loader:
                lr, hr, trans = data
                lr = lr.to(device)
                hr = hr.to(device)

                with torch.inference_mode():
                    sr = self.diffusion_model.super_resolution(lr)
                    sr = torch.clamp(sr, -1, 1)

                    sr = sr.detach().cpu()
                    hr = hr.detach().cpu()
                    sr = dem_data_convert.tensor4D_maxmin_denorm(sr, trans)
                    hr = dem_data_convert.tensor4D_maxmin_denorm(hr, trans)
                    # river miou输入为-1,1 按照各自maxmin归一化
                    sr_,_=dem_data_convert.tensor4D_maxmin_norm(sr)
                    sr_=sr_.to(device)
                    hr_,_=dem_data_convert.tensor4D_maxmin_norm(hr)
                    hr_=hr_.to(device)
                    miou, _, _ = self.river_miou(sr_, hr_)
                    eval_results['miou'].append(miou)
                    eval_res = dem_features.cal_DEM_metric(hr, sr)
                for key, value in eval_res.items():
                    eval_results[key].append(value)
                t.update(1)

        results_avg = {}
        for key, value in eval_results.items():
            _mean = np.mean(value)
            results_avg["test/" + key] = _mean
            #writer.add_scalar("test/" + key, _mean)

        if hasattr(self, "logger"):
            results_str = record.compose_kwargs(**results_avg)
            self.logger.info(f'test results: {results_str}')

        if hasattr(self, "wandb_run"):
            self.wandb_run.log(results_avg, step=epoch)

    def super_resolution(self, lr):
        """

        :param lr: tensor shape (B,C,h,w)
        :return: sr: tensor shape (B,C,H,W)
        """
        self.diffusion_model.eval()
        with torch.inference_mode():
            sr = self.diffusion_model.super_resolution(lr)
            sr = torch.clamp(sr, -1, 1)
        return sr

    def use_ddim_diffusion(self):
        """
        only for sample
        change self.diffusion_model to ddim diffusion model
        :return:
        """
        ddim_config=self.config.get("ddim_config",None)
        if ddim_config is None:
            raise Exception("ddim_config is not defined")
        cls_path=ddim_config['path']
        cls=tools.get_object_from_path_v2(cls_path)
        section_counts=ddim_config['section_counts']
        use_timesteps=space_timesteps(self.timesteps,section_counts)
        spaced_diffusion=cls(
            use_timesteps=use_timesteps,
            gd_object=self.diffusion_model
        )
        self.old_diffusion_model=self.diffusion_model
        # ddim model
        self.diffusion_model=spaced_diffusion.to(self.device)


    def free(self):
        if hasattr(self, "wandb_run"):
            self.wandb_run.finish()
        if hasattr(self, "recorder"):
            self.recorder.free()

    def save_config(self):
        if hasattr(self, "recorder"):
            self.recorder.save_config_to_yaml(self.config)
        else:
            config_save_path=os.path.join(self.output_path, f'{self.start_time}')
            tools.make_dir(config_save_path)
            config_yaml=os.path.join(config_save_path, 'config.yaml')
            record.save_config_to_yaml(self.config,config_yaml)

    def set_model_save_path(self,path=None):
        if path is None:
            if hasattr(self, "recorder"):
                model_save_path = os.path.join(self.recorder.save_path, 'checkpoint')
            else:
                model_save_path = os.path.join(self.output_path, f'{self.start_time}', 'checkpoint')
            tools.make_dir(model_save_path)
            self.model_save_path = model_save_path
        else:
            self.model_save_path = path

    # 保存的是unet模型
    def save_model(self, epoch, loss):
        self.set_model_save_path()
        model_path = os.path.join(self.model_save_path, f'{self.model_name}_{epoch}_{loss:.6f}.pth')
        torch.save(self.denoise_network.state_dict(), model_path)
        if hasattr(self, "logger"):
            self.logger.info(f"denoise_model save path:{model_path}")


    def load_model(self, model, weight_path):
        state_dict=torch.load(weight_path, weights_only=True)
        model.load_state_dict(state_dict)

    def save_checkpoint(self,current_epoch):
        raise NotImplementedError


    def load_checkpoint(self,model,checkpoint_path):
        raise NotImplementedError


    def train_resume(self):
        raise NotImplementedError

    def test_from_ckp_list(self,ckp_list):
        raise NotImplementedError


if __name__ == '__main__':
    pass
