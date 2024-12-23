# Description: This file is used to import all the models in the models directory.
from typing import Literal
from .PINN import PINN
from .FLS import FLS
from .PINNsformer import PINNsformer
from .KAN import KAN
from .DeepONet import DeepONet
from .FNO import FNO1d
from .RBFKAN import RBFKAN
from .fftKAN import fftKAN
from .wavKAN import wavKAN
from .powermlp import PowerMLP

__all__ = ['PINN','FLS','PINNsformer','KAN','DeepONet','FNO1d','RBFKAN','fftKAN','wavKAN','PowerMLP']

ACT = Literal["relu", "tanh","sigmoid", "silu","gelu","leakyrelu"] 
MODELS = Literal["pinn","pinnsformer","kan","fls","rbfkan","fftkan","wavkan","powermlp"]


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
        case "powermlp":
            model = PowerMLP(dim_list=[2,32,16,1],repu_order=3)
    return model
