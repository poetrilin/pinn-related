import os
from typing import Literal
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys 

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import set_seed,get_model

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

set_seed(seed=42)

# 定义泊松方程的右端项 f(x, y)
def f(x, y):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# 定义真实解（用于误差分析），泊松方程的解析解假设为：
def true_solution(x, y):
    return (1 / (2 * torch.pi ** 2)) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)

# 定义损失函数
def loss_function(model, x, y, boundary_x, boundary_y):
    # 内部点的损失
    x.requires_grad_(True)
    y.requires_grad_(True)
    u = model(torch.cat((x, y), dim=1))
    grads = torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_y = grads
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    equation_loss = torch.mean((u_xx + u_yy + f(x, y)) ** 2)

    # 边界点的损失
    boundary_u = model(torch.cat((boundary_x, boundary_y), dim=1))
    boundary_loss = torch.mean(boundary_u ** 2)  # Dirichlet条件：u=0

    return equation_loss +  boundary_loss

# 生成训练数据
def generate_data(N_inside, N_boundary):
    # 内部点
    x = torch.rand(N_inside, 1, requires_grad=True).to(device)
    y = torch.rand(N_inside, 1, requires_grad=True).to(device)
    # 边界点
    boundary_x = torch.cat(
        (torch.zeros(N_boundary, 1), torch.ones(N_boundary, 1), torch.rand(N_boundary, 1), torch.rand(N_boundary, 1))).to(device)
    boundary_y = torch.cat(
        (torch.rand(N_boundary, 1), torch.rand(N_boundary, 1), torch.zeros(N_boundary, 1), torch.ones(N_boundary, 1))).to(device)
    return x, y, boundary_x, boundary_y

# train PINN with Adam optimizer
def train_adam(model,
               x,y,
               boundary_x,
               boundary_y,
               *,
               epochs =30000,
               lr=1e-4,
               verbose = True):
    
    loss_list = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-8)
    for epoch in tqdm(range(1, epochs + 1),total=epochs,desc="Training with Adam"):
        optimizer.zero_grad()
        loss = loss_function(model, x, y, boundary_x, boundary_y)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        loss_list.append(loss.item())
        if epoch % 1000 == 0 and verbose:
            print(f"Adam Epoch {epoch}, Loss: {loss.item():.6e}")
        # if epoch >=50 and epoch % 20 == 0:
        #     torch.save(model.state_dict(),f"trained_models/{model_name}-{epoch}.pth")
    return model,loss_list

# train PINN with LBFGS optimizer
def train_lbfgs(model,
                x,y,
               boundary_x,
               boundary_y,
               *,
               epochs = 11000,
               lr=1e-5,
               verbose = True):
    loss_list = []
    # 模型和优化器
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr,line_search_fn="strong_wolfe")
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0)
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x, y, boundary_x, boundary_y)
        loss.backward()
        return loss
    
    for epoch in tqdm(range(1, epochs + 1),total=epochs,desc="Training with L-BFGS"):
        optimizer.step(closure)
        lr_scheduler.step()
        loss = closure()
        loss_list.append(loss.item())
        if epoch % 1000 == 0 and verbose:
            print(f"LBFGS Epoch {epoch}, Loss: {loss.item():.6e}")
        # if epoch >=50 and epoch % 20 == 0:
        #     torch.save(model.state_dict(),f"trained_models/{model_name}-{epoch}.pth")
    return model,loss_list


def plot_loss(loss_list,save_path = None,log_scale = True):  
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    if log_scale:
        plt.yscale("log")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    if isinstance(save_path,str):
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
# 训练并验证
if __name__ == "__main__":
    act = "tanh".lower()
    model_name = "pinn".lower()
    model = get_model(act=act,model_name = model_name).to(device)
    # save model
    model_save_path = os.path.join(os.getcwd(),"trained_models")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    loss_save_path = os.path.join(os.getcwd(),"img")
    if not os.path.exists(loss_save_path):
        os.makedirs(loss_save_path)
    N_inside = 1000
    N_boundary = 200
    x, y, boundary_x, boundary_y = generate_data(N_inside, N_boundary)
    adam_trained_flag = False
    if adam_trained_flag:
        model.load_state_dict(torch.load(os.path.join(model_save_path,f"{model_name}-adam.pth")))
    else:
        trained_model,loss_list_adam = train_adam( model,
                                                x,y,
                                            boundary_x,
                                            boundary_y  
                                            )
        print(f"Period 1 end, Loss: {loss_list_adam[-1]:.6e}")
        torch.save(model.state_dict(),os.path.join(model_save_path,f"{model_name}-adam.pth"))
        plot_loss(loss_list_adam,save_path = os.path.join(loss_save_path,f"{model_name}-adam_loss.png"))    
    
    
    model, loss_list_lbfgs = train_lbfgs( model,x,y,
                                          boundary_x,
                                          boundary_y,
                                          lr=1e-5
                                          )
    torch.save(model.state_dict(),os.path.join(model_save_path,f"{model_name}.pth"))
    # save loss curve
    
    plot_loss(loss_list_lbfgs,save_path = os.path.join(loss_save_path,f"{model_name}-lbfgs_loss.png"))

    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    