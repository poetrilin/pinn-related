import os
import h5py
import numpy as np
import torch 
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
from models import DeepONet,FNO1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64

class BurgersDataset(Dataset):
    def __init__(self, data_path:str = "Data/1D_Burgers_Sols_Nu0.001.hdf5"):
        super(BurgersDataset, self).__init__()
        f = h5py.File(data_path, 'r')
        self.t_coordinate=  f['t-coordinate'][:-1]
        self.x_coordinate=  f['x-coordinate']
        self.u_itx = torch.tensor(f['tensor'][:1000],dtype=torch.float32)
        self.N = self.u_itx.shape[0]
        self.T_tick = self.u_itx.shape[1]
        self.N_x = self.u_itx.shape[2]
        self.x_coordinate = torch.tensor(self.x_coordinate,dtype=torch.float32)
        self.t_coordinate = torch.tensor(self.t_coordinate,dtype=torch.float32)
        self.u0 = self.u_itx[:,0,:]
        self.u1 = self.u_itx[:,-1,:]
        self.u_itx = self.u_itx[:,1:-1,:]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        return self.u0[idx],self.u_itx[idx],self.u1[idx]
    def get_mesh(self):
        """ return train_mesh,test_mesh 
        shape: (ticks*N_x, 2),train_ticks = T_tick-2=199, test_ticks = 1 ,N_x = 1024
        """
        train_mesh = torch.stack(torch.meshgrid(self.t_coordinate[1:-1],self.x_coordinate),dim=-1).reshape(-1,2)
        test_mesh = torch.stack(torch.meshgrid(self.t_coordinate[-1].unsqueeze(dim=0),self.x_coordinate),dim=-1).reshape(-1,2)
        return train_mesh,test_mesh

def get_data(data_path:str = "Data/1D_Burgers_Sols_Nu0.001.hdf5"):
    """
    @return: loader([u_0(x) , u(x,t),u_1(x)]) ,  train_mesh,test_mesh
    """    
    dataset = BurgersDataset(data_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_mesh,test_mesh = dataset.get_mesh()
    return loader,train_mesh,test_mesh

def loss_fn(y_pred, y_true):
    return torch.mean(torch.square(y_pred - y_true))

def train_step(model, optimizor,dataloader,tx_mesh ,criterion):
    train_running_loss = 0
    for idx, data in enumerate(dataloader):
        # u_0(batch, N_x), u_itx(batch, T_tick, N_x)
        # tx_mesh (ticks*N_x, 2),here ticks = T_tick-2=199, N_x = 1024
        u_0, u_itx , _ = data
        u_0 = u_0.to(device)
        tx_mesh = tx_mesh.to(device)
        u_pred = model(u_0,tx_mesh) # (batch, ticks*N_x)
        u_itx = u_itx.to(device).view(-1,tx_mesh.shape[0]) # (batch, ticks*N_x)
        loss = criterion(u_pred, u_itx)
        optimizor.zero_grad()
        loss.backward()
        optimizor.step()
        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)
    return train_loss

@torch.no_grad()
def test_step(model, dataloader, tx_mesh, criterion):
    test_running_loss = 0
    for idx, data in enumerate(dataloader):
        u_0 , _ , u_1 = data
        u_0 = u_0.to(device) # (batch, N_x)
        u_1 = u_1.to(device) # (batch, N_x)
        tx_mesh = tx_mesh.to(device)
        u_pred  = model(u_0,tx_mesh) # (batch, ticks*N_x)
        loss    = criterion(u_pred, u_1)
        test_running_loss += loss.item()
    test_loss = test_running_loss / (idx + 1)
    return test_loss

def train(model, max_epochs:int, optimizor:torch.optim,data_path:str):
    data_loader,train_mesh,test_mesh = get_data(data_path)
    train_loss_hist = []
    test_loss_hist = []
    for epoch in tqdm(range(max_epochs), desc="Training",total=max_epochs):
        model.train()
        train_loss = train_step(model, optimizor, data_loader, train_mesh, loss_fn)
        train_loss_hist.append(train_loss)
        model.eval()
        test_loss = test_step(model, data_loader, test_mesh, loss_fn)
        test_loss_hist.append(test_loss)

        if epoch>100 and abs(train_loss_hist[-1] - train_loss_hist[-2])< 1e-6:
            print(f"early stop at epoch {epoch}")
            print(f"Epoch {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
            break
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Training Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    return train_loss_hist,test_loss_hist

if __name__ == "__main__":
    model = DeepONet(1024,branch_layers=2,trunk_layers=2,p=16, is_stack=False).to(device)
    optimizor = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_path = os.path.join(os.getcwd(),"Data/1D_Burgers_Sols_Nu0.001.hdf5")
    train_loss,test_loss = train(model, max_epochs=500, optimizor=optimizor,data_path=data_path)
    train_loss = np.array(train_loss)
    np.save("train_loss.npy",train_loss)
    test_loss = np.array(test_loss)
    np.save("test_loss.npy",test_loss)
    
    torch.save(model.state_dict(), "DeepONet.pth")
