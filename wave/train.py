"""  
1-D Wave Problem
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from plotting import plot_loss
from utils import set_seed
from models import get_model
from settings import BETA,SEED

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

set_seed(seed=SEED)

# 定义损失函数
def loss_function(model, x , t , 
                  x_lower , t_lower, 
                  x_left, t_left, 
                  x_right, t_right):
    # 内部点的损失
    x.requires_grad_(True)
    t.requires_grad_(True)
    t_lower.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))      
    grads = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_t = grads
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
    loss_res = torch.mean((u_tt-BETA*u_xx)**2)

    pred_low = model(torch.cat((x_lower, t_lower), dim=1))

    loss_ic = torch.mean( (pred_low - torch.sin(torch.pi*x_lower) 
                          - 0.5*torch.sin(BETA*torch.pi*x_lower) )**2)
    u_t_x0 = torch.autograd.grad(pred_low,t_lower, grad_outputs=torch.ones_like(pred_low),create_graph=True,allow_unused=True)[0]
    loss_ic += torch.mean(u_t_x0**2)
    loss_bc = torch.mean((model(torch.cat((x_left, t_left), dim=1)))**2) + \
                    torch.mean((model(torch.cat((x_right, t_right), dim=1)))**2)

    # 边界点的损失
    return loss_res + 0.5*loss_bc + 0.2*loss_ic

# 生成训练数据
def generate_data(N_inside, N_boundary, x_max = 2*torch.pi, t_max = 1):
    x = torch.rand(N_inside, 1, requires_grad=True).to(device)
    t = torch.rand(N_inside, 1, requires_grad=True).to(device)

    x_lower = x_max *torch.rand(N_boundary, 1).to(device)
    t_lower = torch.zeros(N_boundary, 1).to(device)
    x_left =  torch.zeros(N_boundary, 1).to(device)
    t_left =  t_max * torch.rand(N_boundary, 1).to(device)
    x_right = x_max * torch.ones(N_boundary, 1).to(device) 
    t_right = t_max * torch.rand(N_boundary, 1).to(device)

    return x, t , x_lower , t_lower, x_left, t_left, x_right, t_right

# train PINN with Adam optimizer
def train_adam(model,
               x,y,
                x_lower , t_lower,
                x_left, t_left, 
                x_right, t_right,
               *,
               epochs =20000,
               lr=2e-3,
               verbose = True):
    
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-8)
    for epoch in tqdm(range(1, epochs + 1),total=epochs,desc="Training with Adam"):
        optimizer.zero_grad()
        loss = loss_function(model, x, y, x_lower , t_lower, x_left, t_left, x_right, t_right)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_list.append(loss.item())
        if epoch % 1000 == 0 and verbose:
            print(f"Adam Epoch {epoch}, Loss: {loss.item():.6e}")
        # if epoch >=50 and epoch % 20 == 0:
        #     torch.save(model.state_dict(),f"trained_models/{model_name}-{epoch}.pth")
        if epoch >= 1000 and abs(loss_list[-1]-loss_list[-2])< 1e-16:
            print(f"Early stopping at epoch {epoch}, Loss: {loss.item():.6e}")
            break 
    return model,loss_list

# train PINN with LBFGS optimizer
def train_lbfgs(model,
                x,y,
                 x_lower , t_lower,
                 x_left, t_left, 
                 x_right, t_right,
               *,
               epochs = 200,
               lr=0.1,
               verbose = True):
    loss_list = []
    # 模型和优化器
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,line_search_fn="strong_wolfe")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0)
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x, y,x_lower , t_lower, x_left, t_left, x_right, t_right)
        loss.backward()
        return loss
    
    for epoch in tqdm(range(1, epochs + 1),total=epochs,desc="Training with L-BFGS"):
        optimizer.step(closure)
        lr_scheduler.step()
        loss = closure()
        loss_list.append(loss.item())
        if epoch % 50 == 0 and verbose:
            print(f"LBFGS Epoch {epoch}, Loss: {loss.item():.6e}")
        # if epoch >=50  vand epoch % 20 == 0:
        #     torch.save(model.state_dict(),f"trained_models/{model_name}-{epoch}.pth")
        if epoch >= 50 and abs(loss_list[-1]-loss_list[-2])< 1e-17:
            print(f"Early stopping at epoch {epoch}, Loss: {loss:.6e}")
            break
    return model, loss_list


def plot_loss(loss_list,save_path = None,log_scale = True):  
    if isinstance(loss_list, list) and len(loss_list) > 0:
        plt.plot(loss_list)
        plt.xlabel("Epoch")
        if log_scale:
            plt.yscale("log")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
    elif isinstance(loss_list, dict) and len(loss_list) > 0:
        for key, value in loss_list.items():
            plt.plot(value, label=key)
        plt.xlabel("Epoch")
        if log_scale:
            plt.yscale("log")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
    
    if isinstance(save_path,str):
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

# 训练并验证
if __name__ == "__main__":
    model_name = "kan".lower()
    problem_str = "wave"
    model = get_model(model_name = model_name,input_dim=2,output_dim=1, problem=problem_str).to(device)
    # save model
    model_save_path = os.path.join(os.getcwd(),"trained_models")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    loss_save_path = os.path.join(os.getcwd(),"img")
    if not os.path.exists(loss_save_path):
        os.makedirs(loss_save_path)
    N_inside = 2000
    N_boundary = 400
    x, y, x_lower , t_lower, x_left, t_left, x_right, t_right = generate_data(N_inside, N_boundary)
    model,loss_list_adam = train_adam(  model,
                                        x,y,
                                        x_lower ,  t_lower,
                                        x_left, t_left, 
                                        x_right, t_right,
                                        lr=5e-4,
                                        epochs=30000,  
                                        )
    model, loss_list_lbfgs = train_lbfgs( model,x,y,
                                          x_lower , t_lower, x_left, t_left, x_right, t_right,
                                          lr=1e-5,
                                          epochs = 2000,
                                          )
    
    
    torch.save(model.state_dict(),os.path.join(model_save_path,f"{model_name}.pth")) 
    # save loss curve
    # loss_lists = dict("adam":loss_list_adam,"lbfgs":loss_list_lbfgs) 
    plot_loss(loss_list_adam,save_path = os.path.join(loss_save_path,f"{model_name}-adam_loss.png"))

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    