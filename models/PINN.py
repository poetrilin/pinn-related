import torch
import torch.nn as nn
import numpy as np
import sys 
sys.path.append('..')
from utils import ACT


class SinAct(nn.Module):
    def __init__(self):
            super(SinAct, self).__init__() 

    def forward(self, x):
        return torch.sin(x)

# 定义PINN模型
class PINN(nn.Module):
    def __init__(self,
                 activation:ACT="tanh",
                 d_in:int=2,
                 d_out:int=1,
                 d_hidden:int=20,
                 ):
        super(PINN, self).__init__()
        match activation:
            # to use mathc-case, python version >= 3.10
            case "relu":
                self.act = nn.ReLU()
            case "tanh":
                self.act = nn.Tanh()
            case "sigmoid":
                self.act = nn.Sigmoid()
            case "silu":
                self.act = nn.SiLU()
            case "gelu":
                self.act = nn.GELU()
            case "leakyrelu":    
                self.act = nn.LeakyReLU()
        self.layers = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            SinAct(),
            nn.Linear(d_hidden, d_hidden),
            self.act,
            nn.Linear(d_hidden, d_hidden),
            self.act,
            nn.Linear(d_hidden, d_hidden),
            self.act,
            nn.Linear(d_hidden, d_out)
        )


    def forward(self, x):
        """ 
        x :(N,D), N is the number of samples, D is the dimension of input
        """
        return self.layers(x)

