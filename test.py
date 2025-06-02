import sys
import os
import torch

if __name__ == '__main__':
    t1=torch.tensor(
        [1,2,3,4,5,6,7,8]
    ,dtype=torch.float)
    t2=t1+1

    t1=t1.reshape(1,2,2,2)
    t2=t2.reshape(1,2,2,2)

    t1_mean=torch.mean(t1,dim=(1,2,3))
    pass
