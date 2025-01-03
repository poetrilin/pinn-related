# Description: This file is used to import all the models in the models directory.
from typing import Literal
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

MODELS = Literal["pinn","pinnsformer","kan","fls","rbfkan","fftkan","wavkan","powermlp"]

def get_model(
              model_name:MODELS = "pinn",
              input_dim = 2,
              hidden_dim = 64,
              output_dim = 1,
              problem:str = "poisson"
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
                case "fls":
                    model = PINN(layers=[input_dim,hidden_dim,hidden_dim,hidden_dim,output_dim], is_fls=True)  
                case "pinnsformer":
                    model = PINNsformer(d_in=input_dim,d_out=output_dim,d_hidden=hidden_dim,d_model=32,N=1,heads=2)
                case "kan":
                    model = KAN(layers=[input_dim,16,16,output_dim])
                case "rbfkan":
                    model = RBFKAN(input_dim=2,hidden_dim=64,output_dim=1,num_centers=32,hidden_layers=1)
                case "fftkan":
                    model = fftKAN(inputdim=2,outdim=1,hidden_dim=32,gridsize=5,hidden_layers=1)
                case "wavkan":
                    model = wavKAN(layers_hidden=[2,16,32,32,1])
                case "powermlp":
                    model = PowerMLP(dim_list=[2,16,32,16,1], repu_order= 3)
        case "convection":
            if input_dim is None:
                input_dim = 2
            if output_dim is None:
                output_dim = 1
            match model_name:
                case "pinn":
                    model = PINN(layers=[input_dim,16,32,64,128,64,32,16,output_dim], is_fls=False)
                case "pinnsformer":
                    model = PINNsformer(d_in=input_dim,d_out=output_dim,d_hidden=hidden_dim,d_model=32,N=1,heads=2)
                case "kan":
                    model = KAN(layers=[input_dim,6,14,6,output_dim])
                case "powermlp":
                    model = PowerMLP(dim_list=[2,16,32,16,1], repu_order= 3)
        case "wave":
            if input_dim is None:
                input_dim = 2
            if output_dim is None:
                output_dim = 1
            match model_name:
                case "pinn":
                    model = PINN(layers=[input_dim,16,32,64,32,16,output_dim], is_fls=False)
                case "pinnsformer":
                    model = PINNsformer(d_in=input_dim,d_out=output_dim,d_hidden=hidden_dim,d_model=32,N=1,heads=2)
                case "kan":
                    model = KAN(layers=[input_dim,6,10,6,output_dim])
                case "powermlp":
                    model = PowerMLP(dim_list=[2,16,32,16,1], repu_order= 3)
    return model
