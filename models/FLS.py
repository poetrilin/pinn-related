"""
for high frequency,you can substitute the first layer with this activation function
@ baseline implementation of First Layer Sine
@ paper: Learning in Sinusoidal Spaces with Physics-Informed Neural Networks
@ link: https://arxiv.org/abs/2109.09338
"""

import torch
import torch.nn as nn
import numpy as np
import sys 
sys.path.append('..')
from mytype import ACT


class SinAct(nn.Module):
    def __init__(self):
            super(SinAct, self).__init__() 

    def forward(self, x):
        return torch.sin(x)

# 定义FLS模型
class FLS(nn.Module):
    def __init__(self,
                 activation:ACT="tanh",
                 d_in:int=2,
                 d_out:int=1,
                 d_hidden:int=64,
                 n_layer_hidden:int=3
                 ):
        super(FLS, self).__init__()
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
            *[nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                self.act
            ) for _ in range(n_layer_hidden)],
            nn.Linear(d_hidden, d_out)
        )

    def forward(self, x):
        """ 
        x :(N,D), N is the number of samples, D is the dimension of input
        """
        return self.layers(x)

