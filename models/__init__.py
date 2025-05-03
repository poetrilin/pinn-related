# Description: This file is used to import all the models in the models directory.
from typing import Literal
import torch
from .PINN import PINN
from .PINNsformer import PINNsformer
from .KAN import KAN
from .DeepONet import DeepONet
from .FNO import FNO1d
from .RBFKAN import RBFKAN
from .fftKAN import fftKAN
from .wavKAN import wavKAN
from .powermlp import PowerMLP

__all__ = ['PINN','PINNsformer','KAN','DeepONet','FNO1d','RBFKAN','fftKAN','wavKAN','PowerMLP']

MODELS = Literal["pinn","pinnsformer","kan","fls","rbfkan","fftkan","wavkan","powermlp","fno1d"]

def get_model(
              model_name:MODELS = "pinn",
              input_dim = 2,
              hidden_dim = 64,
              output_dim = 1,
              problem:str = "poisson",
              activation:str =None,
              ):
    match problem:
        case "poisson":
            if input_dim is None:
                input_dim = 2
            if output_dim is None:
                output_dim = 1
            match model_name:
                case "pinn":
                    model = PINN(layers=[input_dim,hidden_dim,hidden_dim,hidden_dim,output_dim], is_fls=False)
                # case "fls":
                #     model = PINN(layers=[input_dim,hidden_dim,hidden_dim,hidden_dim,output_dim], is_fls=True)   
                case "kan":
                    model = KAN(layers=[input_dim,16,16,output_dim])
                # case "rbfkan":
                #     model = RBFKAN(input_dim=2,hidden_dim=64,output_dim=1,num_centers=32,hidden_layers=1)  
                case "powermlp":
                    model = PowerMLP(dim_list=[2,64,64,64,1], repu_order= 3)
        case "convection":
            if input_dim is None:
                input_dim = 2
            if output_dim is None:
                output_dim = 1
            match model_name:
                case "pinn":
                    model = PINN(layers=[input_dim,16,32,64,128,64,32,16,output_dim], is_fls=False)
                case "kan":
                    if activation == "mish":
                        model = KAN(layers=[input_dim,6,12,8,8,output_dim],spline_order=3,base_activation=torch.nn.Mish)
                        # model = KAN(layers=[input_dim,11,14,6,3,output_dim],spline_order=3,base_activation=torch.nn.Mish)
                    else:
                        model = KAN(layers=[input_dim,6,12,8,8,output_dim],spline_order=3)
                case "powermlp":
                    if activation == "mish":
                        model = PowerMLP(dim_list=[input_dim ,32,64,64,32,output_dim], repu_order=3,act="mish")
                    else:
                        model = PowerMLP(dim_list=[input_dim,16,32,64,64,32,16,output_dim], repu_order=3,act = "silu")
        case "wave":
            if input_dim is None:
                input_dim = 2
            if output_dim is None:
                output_dim = 1
            match model_name:
                case "pinn":
                    model = PINN(layers=[input_dim,32,128,128,32,output_dim], is_fls=False)
                case "kan":
                    if activation == "mish":
                        model = KAN(layers=[input_dim,8,16,4,output_dim],base_activation=torch.nn.Mish)
                    else:
                        model = KAN(layers=[input_dim,8,16,4,output_dim])
                case "powermlp":
                    if activation == "mish":
                        model = PowerMLP(dim_list=[2,32,64,64,32,1], repu_order= 3,act="mish")
                    else:
                        model = PowerMLP(dim_list=[2,32,64,64,32,1], repu_order= 3,act = "silu")
    return model
