import os
from typing import Literal
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models import PINN,FLS
from utils import ACT,MODELS,set_seed
from metric import check_relative_error

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
    x = torch.rand(N_inside, 1, requires_grad=True)
    y = torch.rand(N_inside, 1, requires_grad=True)
    # 边界点
    boundary_x = torch.cat(
        (torch.zeros(N_boundary, 1), torch.ones(N_boundary, 1), torch.rand(N_boundary, 1), torch.rand(N_boundary, 1)))
    boundary_y = torch.cat(
        (torch.rand(N_boundary, 1), torch.rand(N_boundary, 1), torch.zeros(N_boundary, 1), torch.ones(N_boundary, 1)))
    return x, y, boundary_x, boundary_y


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
        case "pinnformer":
            raise NotImplementedError("Pinnformer model is not implemented yet")
        case "kan":
            raise NotImplementedError("KAN model is not implemented yet")
    return model
# train  PINN with LBFGS optimizer
def train_lbfgs(model,*,
          epochs = 30,
          lr = 0.1,
          N_inside   = 1000,
          N_boundary = 200,
          verbose    = True):
    x, y, boundary_x, boundary_y = generate_data(N_inside, N_boundary)
    loss_list = []
    # 模型和优化器
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr)
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x, y, boundary_x, boundary_y)
        loss.backward()
        return loss
    # 训练
    for epoch in range(1,epochs+1):
        optimizer.step(closure)
        loss = closure()
        loss_list.append(loss.item())
        if epoch % 5 == 0 & verbose:
            print(f"Epoch {epoch}, Loss: {loss.item():.6e}")
    return model,loss_list




def plot_loss(loss_list,save_path = None):
    
    plt.plot(loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    if isinstance(save_path,str):
        plt.savefig(save_path)
    else:
        plt.show()
# 训练并验证
if __name__ == "__main__":
    act = "tanh".lower()
    model_name = "fls".lower()
    model = get_model(act=act,model_name=model_name)
    trained_model,loss_list = train_lbfgs(model)
    # save model
    model_save_path = os.path.join(os.getcwd(),"trained_models")
    torch.save(trained_model.state_dict(),os.path.join(model_save_path,f"{act}_{model_name}.pth"))
    # save loss curve
    loss_save_path = os.path.join(os.getcwd(),"img")
    plot_loss(loss_list,save_path = os.path.join(loss_save_path,f"{act}_loss.png"))
    # check metrics and plot
    check_relative_error(trained_model,test_points = 100,plot_flag = True)