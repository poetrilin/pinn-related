import time
import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys 
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from plotting import plot_loss
from utils import set_seed
from models import get_model
from settings import nu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using {device}")

set_seed(seed=42)
 
 
def loss_function(model, x , t , 
                  x_lower , t_lower, 
                  x_left, t_left, 
                  x_right, t_right):
    # 内部点的损失
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(torch.cat((x, t), dim=1))      
    grads = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
    u_x, u_t = grads
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    loss_res = torch.mean( (u_t+ u*u_x - nu*u_xx)**2)

    loss_ic = torch.mean((model(torch.cat((x_lower, t_lower), dim=1)) - torch.sin(x_lower))**2)
    loss_bc = torch.mean((model(torch.cat((x_left, t_left), dim=1))**2))+ \
                torch.mean((model(torch.cat((x_right, t_right), dim=1))**2))

    # 边界点的损失
    return loss_res + 0.25*loss_bc + 0.5*loss_ic

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
               x, y,
               boundary_x,
               boundary_y,
               *,
               epochs = 20000,
               lr = 5e-4,
               verbose = True):
    
    loss_list = []
    optimizer =    torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-9)
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
        if epoch >= 1000 and abs(loss_list[-1]-loss_list[-2])< 1e-16:
            print(f"Early stopping at epoch {epoch}, Loss: {loss.item():.6e}")
            break 
    return model,loss_list

# train PINN with LBFGS optimizer
def train_lbfgs(model,
                x,y,
               boundary_x,
               boundary_y,
               *,
               epochs = 500,
               lr=1e-5,
               verbose = True):
    loss_list = []
    # 模型和优化器
    optimizer = torch.optim.LBFGS(model.parameters(), 
                                  lr=lr,
                                  line_search_fn = "strong_wolfe",
                                  tolerance_grad = 1e-11, 
                                  tolerance_change = 1e-13)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=0)
    def closure():
        optimizer.zero_grad()
        loss = loss_function(model, x, y, boundary_x, boundary_y)
        loss.backward()
        return loss
    
    for epoch in tqdm(range(1, epochs + 1), total=epochs, desc="Training with L-BFGS"):
        optimizer.step(closure)
        lr_scheduler.step()
        loss = closure()
        loss_list.append(loss.item())
        if epoch % 100 == 0 and verbose:
            print(f"LBFGS Epoch {epoch}, Loss: {loss:.4e}")
        # if epoch >=50 and epoch % 20 == 0:
        #     torch.save(model.state_dict(),f"trained_models/{model_name}-{epoch}.pth")
        if epoch >= 50 and abs(loss_list[-1]-loss_list[-2])< 1e-17:
            print(f"Early stopping at epoch {epoch}, Loss: {loss:.6e}")
            break 
    return model,loss_list

def get_args():
    parser = argparse.ArgumentParser(description="Train a PINN for Burgers equation")
    parser.add_argument("-m", "--model_name", type=str, default="pinn", help="Model name")
    parser.add_argument("--problem", type=str, default="burgers", help="Problem type")
    return parser.parse_args()

# 训练并验证
if __name__ == "__main__":
    model_name = "pinn".lower()
    problem_str = "burgers"
    model = get_model(model_name = model_name, input_dim=2, output_dim=1, 
                      problem=problem_str 
                      ).to(device)
    print(f"Model: {model_name} ,Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # save model
    model_save_path = os.path.join(__file__,"../trained_models")
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    loss_save_path = os.path.join(__file__,"../img")
    if not os.path.exists(loss_save_path):
        os.makedirs(loss_save_path)
    N_inside = 5000
    N_boundary = 1000
    x, y, boundary_x, boundary_y = generate_data(N_inside, N_boundary)
    adam_trained_flag = True
    time_start = time.time()
    if adam_trained_flag:
        model.load_state_dict(torch.load(os.path.join(model_save_path,f"{model_name}-adam.pth")))
    else:
        trained_model,loss_list_adam = train_adam( model,
                                                x, y,
                                            boundary_x,
                                            boundary_y  
                                            )
        print(f"Period 1 end, Loss: {loss_list_adam[-1]:.4e}")
        torch.save(model.state_dict(),os.path.join(model_save_path,f"{model_name}-adam.pth"))
        plot_loss(loss_list_adam,save_path = os.path.join(loss_save_path,f"{model_name}-adam_loss.png"))    
    
    time_2 = time.time()
    time_adam = time_2 - time_start 
    print(f"Adam training time: {time_adam:.2f}s")
    model, loss_list_lbfgs = train_lbfgs( model,x,y,
                                          boundary_x,
                                          boundary_y,
                                          lr=1e-4
                                          )
    time_end = time.time()
    time_lbfgs = time_end - time_2
    print(f"L-BFGS training time: {time_lbfgs:.2f}s")
    print(f"Total training time: {time_end - time_start:.2f}s")
    torch.save(model.state_dict(), os.path.join(model_save_path,f"{model_name}.pth"))
    # save loss curve
  
    plot_loss(loss_list_lbfgs,save_path = os.path.join(loss_save_path,f"{model_name}-lbfgs_loss.png"))

    
    