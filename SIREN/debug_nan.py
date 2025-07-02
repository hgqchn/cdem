import sys
import os

import numpy as np
import torch
from SIREN import modules




if __name__ == '__main__':

    target_config={
        "out_features": 1,
        "hidden_features": 16,
        "num_hidden_layers": 2,
        "use_pe": False,
        "num_frequencies": 10,
        "use_hsine": True,
    }

    target_net = modules.SimpleMLPNetv1(**target_config)

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sd_pth=r'../mymodel/debug_target.pth'
    sd=torch.load(sd_pth,weights_only=True,map_location='cpu')
    target_net.load_state_dict(sd)
    #target_net.to(device)

    fcb_hsine = target_net.net

    outputs = {}


    def register_hooks(model):
        for name, module in model.named_modules():
            def hook_fn(module, inp, outp, layer_name=name):
                outputs[layer_name] = (inp[0], outp)
            module.register_forward_hook(hook_fn)
    register_hooks(fcb_hsine)


    input=torch.tensor([[-0.984375, -0.984375]],dtype=torch.float32)
    output= fcb_hsine(input)

    pass
