import torch
import torch.nn as nn
import torch.nn.init as init

class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, alpha=0.5):
        super(RBFKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.alpha = alpha

        self.centers = nn.Parameter(torch.empty(num_centers, input_dim))
        init.xavier_uniform_(self.centers)

        self.weights = nn.Parameter(torch.empty(num_centers, output_dim))
        init.xavier_uniform_(self.weights)

    def gaussian_rbf(self, distances):
        return torch.exp(-self.alpha * distances ** 2)

    def forward(self, x):
        distances = torch.cdist(x, self.centers)
        basis_values = self.gaussian_rbf(distances)
        output = torch.sum(basis_values.unsqueeze(2) * self.weights.unsqueeze(0), dim=1)
        return output
    
class RBFKAN(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers=64, hidden_dim=64,hidden_layers=1):
        super(RBFKAN, self).__init__()
        
        self.inlayer = RBFKANLayer(input_dim, hidden_dim, num_centers)
        self.hidden_layers = nn.ModuleList([RBFKANLayer(hidden_dim, hidden_dim, num_centers) for _ in range(hidden_layers)])
        self.outlayer = RBFKANLayer(hidden_dim, output_dim,num_centers)
        self.act = torch.tanh
    def forward(self, x):
        x = self.inlayer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.act(x)
        x = self.outlayer(x)
        return x