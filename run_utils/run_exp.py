import sys
import os

import logging
from utils import record,tools
from run_utils import base_runner
if __name__ == '__main__':


    exp_config=tools.get_path_from_root('run/config_exp.yaml')
    config=record.get_config_from_yaml(exp_config,True)

    runner=base_runner.BaseRunner(config)
    runner.init_recorder(logger=logging.getLogger(__name__))
    runner.init_wandb(project="run_exp")

    runner.train()
    runner.test_step()

    runner.free()
    pass
