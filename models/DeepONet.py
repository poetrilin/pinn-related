import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.autograd import grad

# Define the neural net
class MLP(nn.Module):
    def __init__(self, 
                 layers, 
                 activation=nn.ReLU(),
                 input_dim=2, 
                 output_dim=1,
                 hidden_dim=50):
        super(MLP, self).__init__()
        self.layers =  nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(layers-2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# Define the model
class DeepONet(nn.Module):
    def __init__(self,N_x,branch_layers=2,trunk_layers=2, p=16, is_stack=False):
        """ default unstacked model """
        super(DeepONet, self).__init__()
        self.stack = is_stack
        self.branch_net = MLP(branch_layers, input_dim= N_x, output_dim= p)
        self.trunk_net  = MLP(trunk_layers,  input_dim= 2  , output_dim= p)

    def forward(self, u, y):
        B = self.branch_net(u) # (N, p), u = (N, N_x)
        T = self.trunk_net(y)  # (XT_mesh, p), y = (N, d),here d=2
        outputs = torch.matmul(B, T.T) # (N, XT_mesh)
        return outputs
    
    