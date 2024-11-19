import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models import PINN,FLS,PINNsformer,KAN
from utils import get_n_paras



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
    # if plot_flag:
    #     # 将张量移动到 CPU 并转换为 NumPy 数组
    #     X_mesh_cpu = X_mesh.cpu().numpy()
    #     Y_mesh_cpu = Y_mesh.cpu().numpy()
    #     u_pred_cpu = u_pred.cpu().numpy().reshape(test_points, test_points)
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.pcolormesh(X_mesh_cpu,Y_mesh_cpu, u_pred.reshape(test_points, test_points), shading="auto")
    #     plt.colorbar()
    #     plt.title("Predicted Solution")
    #     plt.subplot(1, 2, 2)
    #     plt.pcolormesh(X_mesh_cpu,Y_mesh_cpu, u_true.reshape(test_points, test_points), shading="auto")
    #     plt.colorbar()
    #     plt.title("True Solution")
    #     plt.savefig("pred_solution.png")

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
    
    model_path = os.path.join(os.getcwd(),"trained_models/kan-50.pth")
    # model = FLS(d_in=2,d_out=1,d_hidden=64,n_layer_hidden=3,activation="tanh")
    # model = PINNsformer(d_in = 2,d_out = 1, d_hidden=64, d_model=32, N=1, heads=2)

    model = KAN(input_dim=2,hidden_dim=64,output_dim=1,num_layers=1)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    check_relative_error(model, test_points=100, plot_flag=True)
    print(f"rMAE: {rMAE(100, model):.4e}")
    print(f"rRMSE: {rRMSE(100, model):.4e}")
    print(f"Number of parameters: {get_n_paras(model)}")
    