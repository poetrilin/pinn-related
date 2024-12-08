# Description: This file is used to import all the models in the models directory.
from .PINN import PINN
from .FLS import FLS
from .PINNsformer import PINNsformer
from .KAN import KAN
from .DeepONet import DeepONet
from .FNO import FNO1d

__all__ = ['PINN','FLS','PINNsformer','KAN','DeepONet','FNO1d']