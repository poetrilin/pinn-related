import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def true_solution(x, y):
    return (1 / (2 * torch.pi ** 2)) * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
# 检查解的相对误差
def check_relative_error(model,test_points = 100,plot_flag = False):
    # 测试点
    
    x_test = torch.linspace(0, 1, test_points).view(-1, 1)
    y_test = torch.linspace(0, 1, test_points).view(-1, 1)
    X_mesh, Y_mesh = torch.meshgrid(x_test.squeeze(), y_test.squeeze(), indexing="ij")
    X, Y = X_mesh.reshape(-1, 1), Y_mesh.reshape(-1, 1)

    # 模型预测值和真实值
    u_pred = model(torch.cat((X, Y), dim=1)).detach()
    u_true = true_solution(X, Y)

    # 相对误差计算
    relative_error = torch.norm(u_pred - u_true) / torch.norm(u_true)
    print(f"Relative Error: {relative_error.item():.4e}")
    if plot_flag:
        # 画图
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.pcolormesh(X_mesh,Y_mesh, u_pred.reshape(test_points, test_points), shading="auto")
        plt.colorbar()
        plt.title("Predicted Solution")
        plt.subplot(1, 2, 2)
        plt.pcolormesh(X_mesh,Y_mesh, u_true.reshape(test_points, test_points), shading="auto")
        plt.colorbar()
        plt.title("True Solution")
        plt.savefig("pred_solution.png")
