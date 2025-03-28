from typing import List 
import torch.nn as nn 
import sys 
sys.path.append('..')
from utils import check_and_convert_to_int

class RePU(nn.Module):
    def __init__(self, n):
        super(RePU, self).__init__()
        self.n = n
        self.relu = nn.ReLU6()
    
    def forward(self, x):
        return self.relu(x) ** self.n

class ResAct(nn.Module):
    def __init__(self, input_dim, output_dim,act = "silu"):
        super(ResAct, self).__init__()
        if act == "silu":
            self.base_act = nn.SiLU()
        elif act == "mish":
            self.base_act = nn.Mish()
        else:
            raise NotImplementedError(f"Activation function {act} is not implemented.")
        self.fc = nn.Linear(input_dim, output_dim, bias=False) 
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)

    # return: out = W base_act(x) 
    def forward(self, x):
        out = self.base_act(x)
        out = self.fc(out)
        return out

class ResRePUBlock(nn.Module):
    def __init__(self, input_dim, output_dim, repu_order, res=True,act="silu"): 
        super(ResRePUBlock, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.repu = RePU(repu_order)
        if res:
            self.res = ResAct(input_dim, output_dim,act=act)

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)
        if hasattr(self, 'res'):
            self.res.init_weights()
    
    def forward(self, x):
    #  return: W_1 Act + RePU(W_2 x + bias)
        residual = x
        out = self.fc(x)
        out = self.repu(out)
        
        if hasattr(self, 'res'):
            residual = self.res(residual)
            out += residual
        return out

    
class PowerMLP(nn.Module):
    def __init__(self, dim_list:List, repu_order = 3, res=True, act="silu"):
        super(PowerMLP, self).__init__()
        dim_list = check_and_convert_to_int(dim_list)
        assert len(dim_list) > 2, "Dimension list is too short to construct a RRN!"

        res_block_list = []
        for i, dim in enumerate(dim_list[:-2]):
            res_block = ResRePUBlock(dim_list[i], dim_list[i+1], repu_order, res=res,act= act)
            res_block_list.append(res_block)
        self.res_layers = nn.ModuleList(res_block_list)
            
        self.fc = nn.Linear(dim_list[-2], dim_list[-1])

    def init_weights(self):
        for res_layer in self.res_layers:
            res_layer.init_weights()
    
    def forward(self, x):
        for res_layer in self.res_layers:
            x = res_layer(x)
        x = self.fc(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    