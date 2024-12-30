from typing import List
import torch
import torch.nn as nn
import numpy as np
import sys 
sys.path.append('..')

class SinAct(nn.Module):
    """
    for high frequency, you can substitute the first layer with this activation function
    @ implementation of First Layer Sine
    @ paper: Learning in Sinusoidal Spaces with Physics-Informed Neural Networks
    @ link: https://arxiv.org/abs/2109.09338
    """
    def __init__(self):
            super(SinAct, self).__init__() 

    def forward(self, x):
        return torch.sin(x)

# 定义PINN模型
class PINN(nn.Module):
    def __init__(self,
                 layers:List[int],
                 is_fls:bool = False,
                 ):
        super(PINN, self).__init__()
        self.act = nn.Tanh()
        self.layers = nn.Sequential(
            nn.Linear(layers[0], layers[1]),
            is_fls and SinAct() or self.act )
        for i in range(1, len(layers) - 2):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.layers.append(self.act)
        self.layers.append(nn.Linear(layers[-2], layers[-1]))

    def forward(self, x):
        """ 
        x :(N,D), N is the number of samples, D is the dimension of input
        """
        return self.layers(x)

if __name__ == '__main__':
    model = PINN([2, 20, 20, 20, 20, 20, 20, 1],is_fls = True)
    print(model)