import numpy as np
import torch
import torch.nn as nn
from typing import Literal
from models import PINN,FLS,PINNsformer,KAN,RBFKAN,fftKAN,wavKAN
from mytype import ACT,MODELS


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_model(act:ACT = "tanh",
              model_name:MODELS = "pinn",
              input_dim = 2,
              hidden_dim = 64,
              output_dim = 1,
              ):
    match model_name:
        case "pinn":
            model = PINN(activation=act,d_in=input_dim,d_out=output_dim,d_hidden=hidden_dim)
        case "fls":
            model = FLS(activation=act,d_in=input_dim,d_out=output_dim,d_hidden=hidden_dim)    
        case "pinnsformer":
            model = PINNsformer(d_in=input_dim,d_out=output_dim,d_hidden=hidden_dim,d_model=32,N=1,heads=2)
        case "kan":
            model = KAN(input_dim=2,hidden_dim=64,output_dim=1,num_layers=1)
        case "rbfkan":
            model = RBFKAN(input_dim=2,hidden_dim=64,output_dim=1,num_centers=32,hidden_layers=1)
        case "fftkan":
            model = fftKAN(inputdim=2,outdim=1,hidden_dim=32,gridsize=5,hidden_layers=1)
        case "wavkan":
            model = wavKAN(layers_hidden=[2,64,32,1])
    return model



def get_data(x_range, y_range, x_num, y_num):
    x = np.linspace(x_range[0], x_range[1], x_num)
    t = np.linspace(y_range[0], y_range[1], y_num)

    x_mesh, t_mesh = np.meshgrid(x,t)
    data = np.concatenate((np.expand_dims(x_mesh, -1), np.expand_dims(t_mesh, -1)), axis=-1) 
    
    b_left = data[0,:,:] 
    b_right = data[-1,:,:]
    b_upper = data[:,-1,:]
    b_lower = data[:,0,:]
    res = data.reshape(-1,2)
    return res, b_left, b_right, b_upper, b_lower



def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:,i,-1] += step*i
    return src


def get_n_paras(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)