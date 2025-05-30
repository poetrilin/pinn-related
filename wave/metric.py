import math
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_n_paras
from models import get_model
from settings import BETA

def true_solution(x, y):
    # u(x,t)=sin(πx)cos(π \sqrt(β) t)+ 0.5sin(βπx)cos(β\sqrt(β) πt)
    return ( torch.sin(torch.pi*x)*torch.cos(math.sqrt(BETA)*torch.pi*y) + \
                0.5*torch.sin(BETA*torch.pi*x)*torch.cos(BETA*math.sqrt(BETA)*torch.pi*y) )

def sampling(nx:int,ny:int):
    x = torch.linspace(0, 1, nx).view(-1, 1)
    y = torch.linspace(0, 1, ny).view(-1, 1)
    X_mesh, Y_mesh = torch.meshgrid(x.squeeze(), y.squeeze(),indexing="ij")
    return X_mesh, Y_mesh

# 检查解的相对误差
def check_relative_error(model,test_points = 100,plot_flag = False,save_path ="./img/result.png" ):
    X_mesh, Y_mesh = sampling(test_points, test_points)
    X, Y = X_mesh.reshape(-1, 1), Y_mesh.reshape(-1, 1)
    u_pred = model(torch.cat((X, Y), dim=1)).detach()
    u_true = true_solution(X, Y)
    # 相对误差计算
    rMAE = torch.mean(torch.abs(u_pred - u_true)) / torch.mean(torch.abs(u_true))
    rRMSE = torch.norm(u_pred - u_true) / torch.norm(u_true)
    print(f"Relative MAE:   {rMAE.item():.4e}")
    print(f"Relative RMSE: {rRMSE.item():.4e}")
    if plot_flag == True:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        CS0 = ax[0].contourf(X_mesh, Y_mesh, u_true.reshape(test_points, test_points).cpu().numpy(), cmap="viridis",)
        ax[0].set_title("True Solution")
        cbar0 = fig.colorbar(CS0, ax=ax[0])
        CS1 = ax[1].contourf(X_mesh, Y_mesh, u_pred.reshape(test_points, test_points).cpu().numpy(), cmap="viridis")
        ax[1].set_title("Predicted Solution")
        cbar1 = fig.colorbar(CS1, ax=ax[1])
        CS2= ax[2].contourf(X_mesh, Y_mesh, (u_pred - u_true).reshape(test_points, test_points).cpu().numpy(), cmap="viridis")
        ax[2].set_title("Error")
        cbar2 = fig.colorbar(CS2, ax=ax[2]) 
        plt.savefig(save_path)
    return rMAE.item(),rRMSE.item()


if __name__ == "__main__":
    model_name = "powermlp"
    problem_str = "wave" 
    act = "silu"
    model_path = os.path.join(os.getcwd(),f"trained_models/{model_name}-{act}-beta-{BETA}.pth")
    model = get_model(model_name=model_name,input_dim=2,output_dim=1, problem=problem_str,activation=act)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"1D Wave, Model: {model_name}-{act}-beta-{BETA}")
    rMAE, rRMSE =check_relative_error(model, test_points=100, plot_flag=True,save_path=f"./img/{model_name}-{act}-beta-{BETA}-result.png")
    results_path = os.path.join(os.path.dirname(__file__), "./results")
    with open(os.path.join(results_path,f"res.txt"),"a") as f:
        f.write(f"{model_name} relative error:\n") 
        f.write(f"Relative MAE:   {rMAE:.4e}\n")
        f.write(f"Relative RMSE: {rRMSE:.4e}\n")
        f.write(f"Number of parameters: {get_n_paras(model)}\n")
    