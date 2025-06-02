import os
import sys
import csv
import logging
from logging import Filter

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
import utils

import torch
from omegaconf import OmegaConf,DictConfig


class Recorder():
    """
    可配合hydra或者独立使用
    """
    def __init__(self, logger=None,output_path="./",save_with_time=True,extra_name=None,outfile=True,use_tensorboard=True):
        self.current_time = utils.get_current_time()
        #self.cwd=os.getcwd()
        self.output_path = output_path
        self.tensorboard_writer=None
        self.use_tensorboard=use_tensorboard

        if not save_with_time:
            if extra_name:
                self.save_path = os.path.join(self.output_path, extra_name)
            self.tensorboard_path = os.path.join(self.save_path,"tensorboard",self.current_time)
        else:
            if extra_name:
                self.save_path = os.path.join(self.output_path, self.current_time+" "+extra_name)
            else:
                self.save_path = os.path.join(self.output_path, self.current_time)
            self.tensorboard_path = os.path.join(self.save_path, "tensorboard")

        self.formatter=logging.Formatter("[%(asctime)s][%(name)s][%(filename)s][%(levelname)s] - %(message)s")

        if outfile:
            utils.make_dir(self.save_path)
            if use_tensorboard:
                utils.make_dir(self.tensorboard_path)
                self.tensorboard_writer = SummaryWriter(log_dir=self.tensorboard_path)


        if logger is not None:
            self.logger = logger
        else:
            self.logger=logging.getLogger(__name__)
        if not self.logger.hasHandlers():
            self.logger.addHandler(self.get_console_handler())
            if outfile:
                self.logger.addHandler(self.get_file_handler(self.save_path))
            self.logger.setLevel(logging.INFO)
            #self.logger.info("Logger initialized, file and console handlers added")
        else:
            #self.logger.info("Logger initialized, handlers already exist")
            pass


        self.logger.info(f"outfile: {outfile}, save path: {self.save_path}")


    def get_file_handler(self, log_path, level=logging.INFO):

        file_path=os.path.join(log_path, "log.txt")
        filehandler = logging.FileHandler(file_path, encoding="utf-8")
        filehandler.setFormatter(self.formatter)
        filehandler.setLevel(level)
        return filehandler
    def get_console_handler(self,level=logging.INFO):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        console_handler.setLevel(level)  # 设置控制台输出日志的级别
        return console_handler

    def get_logger(self):
        return self.logger
    def get_writer(self):

        if self.tensorboard_writer:
            return self.tensorboard_writer
        else:
            #raise ValueError("use tensorboard=False")
            return None

    def save_csv(self,dict,csv_path=None,filename="results.csv"):
        """
        将值为相同长度的列表的字典保存为csv文件，第一行表头为键值，其余各行为数据。
        :param dict:
        :param filename:
        :return:
        """
        # 确保所有的列表长度相同
        keys = dict.keys()
        rows = zip(
            *[dict[key] if len(dict[key]) > 0 else [''] * max(map(len, dict.values())) for key in
              dict])
        if csv_path is not None:
            utils.make_dir(csv_path)
            save_file = os.path.join(csv_path, filename)
        else:
            save_file=os.path.join(self.save_path, filename)
        with open(save_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            writer.writerows(rows)
        abs_path=os.path.abspath(save_file)
        self.logger.info("csv file saved, path is {}".format(abs_path))
        return abs_path

    def save_config_to_yaml(self,config_dict: dict, file_name="config"):
        # 将字典转换为 OmegaConf 对象
        cfg = OmegaConf.create(config_dict)
        file_path = os.path.join(self.save_path, f"{file_name}.yaml")
        # 保存到 YAML
        OmegaConf.save(config=cfg, f=file_path)
        self.logger.info("config yaml file saved, path is {}".format(file_path))

    def free(self):

        for handles in self.logger.handlers:
            handles.close()
            self.logger.removeHandler(handles)
        logging.shutdown()

        if self.tensorboard_writer is not None:
            #self.logger.info("tensorboard writer closed")
            self.tensorboard_writer.flush()

def save_config_to_yaml(config_dict: dict, file_path: str) -> None:
    """
    将字典类型的配置信息保存为 YAML 文件。

    参数:
        config_dict (dict): 要保存的配置字典。
        file_path (str): YAML 文件的完整保存路径。
    """
    # 将字典转换为 OmegaConf 对象
    cfg = OmegaConf.create(config_dict)
    # 保存到 YAML
    OmegaConf.save(config=cfg, f=file_path)

def get_config_from_yaml(yaml_file,return_dict=True):
    with open(yaml_file, 'r') as f:
        config = OmegaConf.load(f)
    return config if not return_dict else OmegaConf.to_container(config,resolve=True, throw_on_missing=True)

def save_checkpoint(model, optimizer, scheduler, epoch, results_dict, best_epoch, save_path='./'):

    save_file = os.path.join(save_path, f"{epoch}_checkpoint.pth")
    if scheduler:
        scheduler_sd = scheduler.state_dict()
    else:
        scheduler_sd = None
    save_dict = {
        "model_sd": model.state_dict(),
        "optimizer_sd": optimizer.state_dict(),
        "scheduler_sd": scheduler_sd,
        "process_dict": results_dict,
        "epoch": epoch,
        "best_epoch": best_epoch
    }
    torch.save(save_dict, save_file)

#logging Filter
class LoggingFilter(Filter):
    def __init__(self):
        super().__init__()
        self.func_name=""
        self.level="INFO"
        self.filtermsg=""
    def get_funcfilter(self,func_name):
        self.func_name=func_name
        return self.funcfilter
    def funcfilter(self, record):
        return not record.funcName == self.func_name

    def get_levelfilter(self,level):
        self.level=level
        return self.levelfilter
    def levelfilter(self, record):
        return record.levelname == self.level

    def get_messagefilter(self,str_filt):
        self.filtermsg=str_filt
        return self.msgfilter
    def msgfilter(self, record):
        return self.filtermsg not in record.msg
    # 不输出包含 Bad 字段的日志
    def function_message_filter(self,record):
        return 'Bad' not in record.msg

def save_dict_csv_v1(data_dict,save_file="./data.csv"):
    """
    将值为相同长度的列表的字典保存为csv文件，第一行表头为键值，其余各行为数据。每行代表一轮训练的结果。
    :param data_dict: value is list of same length
    :param save_file: csv file path
    :return: csv file absolute path
    """
    # 确保所有的列表长度相同
    keys=data_dict.keys()
    rows = zip(
        *[data_dict[key] if len(data_dict[key]) > 0 else [''] * max(map(len, data_dict.values())) for key in data_dict])
    with open(save_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        writer.writerows(rows)
    return os.path.abspath(save_file)

def save_dict_csv_pd(data_dict,save_file="./data.csv"):
    # 元素不能为空
    df=pd.DataFrame(data_dict)
    df.to_csv(save_file, index=False)
    return os.path.abspath(save_file)

def load_dict_from_csv(csv_file):
    data_dict=pd.read_csv(csv_file)
    data_dict=data_dict.to_dict('list')
    return data_dict

def compose_multi_csv(csv_file,add_dict,sort_col=""):
    df=pd.read_csv(csv_file)
    add_df = pd.DataFrame(add_dict)
    if sort_col:
        df=df.sort_values(by=sort_col)
        add_df=add_df.sort_values(by=sort_col)
    data_dict=df.to_dict("list")
    add_dict=add_df.to_dict("list")

    for add_key,value in add_dict.items():
        if add_key not in data_dict.keys():
            data_dict[add_key]=value
    df=pd.DataFrame(data_dict)
    df.to_csv(csv_file, index=False)

def compose_kwargs(**kwargs):
    """
    compose keyword arguments to a string
    :param kwargs:
    :return:
    """
    str_res=""
    for key, value in kwargs.items():
            str_res=str_res+(f"{key}:{str(value)}")+" "*4
    return str_res

def custom_logger(logger_name=__name__,log_filepath=None):
    logger=logging.getLogger(logger_name)
    if not logger.hasHandlers():
        formatter = logging.Formatter("[%(asctime)s][%(name)s][%(filename)s][%(levelname)s] - %(message)s")
        logger.setLevel(logging.INFO)
        if log_filepath:
            filehandler = logging.FileHandler(log_filepath, encoding="utf-8")
            filehandler.setFormatter(formatter)
            filehandler.setLevel(logging.INFO)
            logger.addHandler(filehandler)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    return logger

def add_stream_handler(logger):
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(filename)s][%(levelname)s] - %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)



if __name__ == "__main__":

    recorder=Recorder(logger=logging.getLogger(__name__),
                             output_path=r'D:\codes\diffusion_dem_sr\output\test')


    # save_pth='../test.log'
    # data = {"name": "Alice", "age": 30, "city": "New York"}
    # log_sth(save_pth,**data)
    # csv_file='./test.csv'
    # log_file='./log.log'
    # data={"loss1":[1,2,3,4,5],"loss2":[1,3,5,7,9],"loss3":[2,4,6,8,10]}
    # save_dict_csv(csv_file,data)
    # log_dict(log_file,**data)
    #formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # formatter = logging.Formatter("%(filename)s- %(message)s")
    # logger=logging.getLogger(__name__)
    # #logger.setLevel(logging.INFO)
    # # # filehandler = logging.FileHandler('./log.txt', encoding="utf-8")
    # # # filehandler.setFormatter(formatter)
    # # # logger.addHandler(filehandler)
    # streamhandler = logging.StreamHandler(sys.stdout)
    # streamhandler.setFormatter(formatter)
    # streamhandler.setLevel(logging.INFO)
    # logger.addHandler(streamhandler)
    # # filter=LoggingFilter()
    # # msgfilter=filter.get_messagefilter("test")
    # # logger.addFilter(msgfilter)
    # logger.info("This is a test message")
    # logger.debug("This is a debug message")
    # logger.warning("This is a warning message")
    pass
