"""  
For data-driven models, DeepONet/FNO as typical examples.
"""
import os
import h5py
import numpy as np
import torch 
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import  FNO1d
from plotting import plot_loss
from utils import set_seed, get_n_paras

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

class BurgersDataset(Dataset):
    def __init__(self, data_path:str = "./data/1D_Burgers_Sols_Nu0.01.hdf5"):
        """ 
         
        """
        super(BurgersDataset, self).__init__()
        f = h5py.File(data_path, 'r')
        # self.t_coordinate=  f['t-coordinate'][:-1]
        # self.x_coordinate=  f['x-coordinate']
        self.u_itx = torch.tensor(f['tensor'][:1000],dtype=torch.float32)
        self.N = self.u_itx.shape[0]
        # self.T_tick = self.u_itx.shape[1]
        self.N_x = self.u_itx.shape[2] 
        # self.u0 = self.u_itx[:,0,:]
        self.u1 = self.u_itx[:,-1,:] # test
        self.u0 = self.u_itx[:,0,:]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.u0[idx],self.u1[idx]
    # def get_mesh(self):
    #     """ return train_mesh,test_mesh 
    #     shape: (ticks*N_x, 2),train_ticks = T_tick-1=200 , test_ticks = 1 ,N_x = 1024
    #     """
    #     train_mesh = torch.stack(torch.meshgrid(self.t_coordinate[0:-1],self.x_coordinate),dim=-1).reshape(-1,2)
    #     test_mesh = torch.stack(torch.meshgrid(self.t_coordinate[-1].unsqueeze(dim=0),self.x_coordinate),dim=-1).reshape(-1,2)
    #     return train_mesh,test_mesh

def get_loader(data_path:str = "data/1D_Burgers_Sols_Nu0.001.hdf5",train_ratio:float = 0.8):
    """
    @return: loader([u_0(x) , u(x,t),u_1(x)]) ,  train_mesh,test_mesh
    """    
    dataset = BurgersDataset(data_path)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, test_loader

def loss_fn(y_pred, y_true):
    return torch.mean(torch.square(y_pred - y_true))

def train_step(model, optimizor,dataloader ,criterion):
    train_running_loss = 0
    for idx, data in enumerate(dataloader): 
        u0 , u1 = data 
        u0 = u0.to(device).unsqueeze(dim=-1)  
        u1 = u1.to(device) # (batch, N_x)
        u_pred = model(u0) # (batch, N_x, 1)
        loss = criterion(u_pred.squeeze(dim=-1)
                         , u1)
        optimizor.zero_grad()
        loss.backward()
        optimizor.step()
        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)
    return train_loss

@torch.no_grad()
def test_step(model, dataloader, criterion):
    test_running_loss = 0
    for idx, data in enumerate(dataloader):
        u_0 , u_1 = data
        u_0 = u_0.to(device) # (batch, N_x)
        u_1 = u_1.to(device) # (batch, N_x)
        u_pred  = model(u_0)
        loss    = criterion(u_pred, u_1)
        test_running_loss += loss.item()
    test_loss = test_running_loss / (idx + 1)
    return test_loss

def train(model, max_epochs:int, optimizor:torch.optim, train_loader:DataLoader,test_loader:DataLoader):  
    train_loss_hist = []
    test_loss_hist = []
    for epoch in tqdm(range(max_epochs), desc="Training",total=max_epochs):
        model.train()
        train_loss = train_step(model, optimizor, train_loader, loss_fn)
        train_loss_hist.append(train_loss)
        model.eval()
        test_loss = test_step(model, test_loader, loss_fn)
        test_loss_hist.append(test_loss)

        if epoch>100 and abs(train_loss_hist[-1] - train_loss_hist[-2])< 1e-8:
            print(f"early stop at epoch {epoch}")
            print(f"Epoch {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            break
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    return train_loss_hist,test_loss_hist

if __name__ == "__main__":
    modes = 16
    widths = 32
    learning_rate = 1e-3
    model = FNO1d(modes, widths).to(device)
    print(f"Model has {get_n_paras(model)} parameters")
    optimizor = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_path = os.path.join(__file__,"../data/1D_Burgers_Sols_Nu0.01.hdf5")
    train_loader, test_loader = get_loader(data_path)
    train_loss,test_loss = train( model, max_epochs=500, optimizor=optimizor,\
                                  train_loader=train_loader, test_loader=test_loader)
    plot_loss(train_loss,save_path="./imgs/train_loss.png")
    save_dir= os.path.join(__file__,"../models/")
    if save_dir is None:
        os.mkdir( save_dir)
    torch.save(model.state_dict(),  os.path.join(save_dir,"FNO1d.pth"))
