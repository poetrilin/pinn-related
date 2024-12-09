import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import PINN,FLS,PINNsformer,KAN,RBFKAN,fftKAN
from utils import get_n_paras, ACT,MODELS,set_seed,get_model

def true_solution(x, y):
    return (1 / (2 * torch.pi ** 2)) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
def sampling(nx:int,ny:int):
    x = torch.linspace(0, 1, nx).view(-1, 1)
    y = torch.linspace(0, 1, ny).view(-1, 1)
    X_mesh, Y_mesh = torch.meshgrid(x.squeeze(), y.squeeze(),indexing="ij")
    return X_mesh, Y_mesh

# 检查解的相对误差
def check_relative_error(model,test_points = 100,plot_flag = False):
    # 测试点
    X_mesh, Y_mesh = sampling(test_points, test_points)
    # X_mesh, Y_mesh = X_mesh.to(device), Y_mesh.to(device)
    X, Y = X_mesh.reshape(-1, 1), Y_mesh.reshape(-1, 1)
    # 模型预测值和真实值
    u_pred = model(torch.cat((X, Y), dim=1)).detach()
    u_true = true_solution(X, Y)

    # 相对误差计算
    relative_error = torch.norm(u_pred - u_true) / torch.norm(u_true)
    print(f"Relative Error: {relative_error.item():.4e}")
    

#rMAE
def rMAE(test_points, model):
    X_mesh, Y_mesh = sampling(test_points, test_points)
    # X_mesh, Y_mesh = X_mesh.to(device), Y_mesh.to(device)
    X, Y = X_mesh.reshape(-1, 1), Y_mesh.reshape(-1, 1)
    
    u_pred = model(torch.cat((X, Y), dim=1)).detach()
    u_true = true_solution(X, Y)
    return torch.mean(torch.abs(u_pred - u_true)) / torch.mean(torch.abs(u_true))


#rMSE
def rRMSE(test_points, model):
    X_mesh, Y_mesh = sampling(test_points, test_points)
    # X_mesh, Y_mesh = X_mesh.to(device), Y_mesh.to(device)
    X, Y = X_mesh.reshape(-1, 1), Y_mesh.reshape(-1, 1)
    
    u_pred = model(torch.cat((X, Y), dim=1)).detach()
    u_true = true_solution(X, Y)
    return torch.sqrt(torch.mean((u_pred - u_true) ** 2)) / torch.sqrt(torch.mean(u_true ** 2))

def test(model_name="kan"):
    for num in [i*10 for i in range(3,11)]:
        model_path = os.path.join(os.getcwd(),f"trained_models/{model_name}-{num}.pth")
        model = KAN(input_dim=2,hidden_dim=64,output_dim=1,num_layers=1)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print(f"for kan trained for {num} epochs:")
        check_relative_error(model, test_points=100, plot_flag=True)
        print(f"rMAE: {rMAE(100, model):.4e}")
        print(f"rRMSE: {rRMSE(100, model):.4e}")
        print('-'*30)


if __name__ == "__main__":
    model_name = "wavkan"
    model_path = os.path.join(os.getcwd(),f"trained_models/{model_name}.pth")
    model = get_model(model_name=model_name)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    check_relative_error(model, test_points=100, plot_flag=True)
    print(f"rMAE: {rMAE(100, model):.4e}")
    print(f"rRMSE: {rRMSE(100, model):.4e}")
    print(f"Number of parameters: {get_n_paras(model)}")
    