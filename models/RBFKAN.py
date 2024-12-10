import torch
import torch.nn as nn
import torch.nn.init as init

class RBFKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_centers, alpha=1.0):
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
    def __init__(self, input_dim, output_dim, num_centers, hidden_dim=None):
        super(RBFKAN, self).__init__()
        if hidden_dim is None:
            hidden_dim = input_dim*2+1
        self.layer1 = RBFKANLayer(input_dim, hidden_dim, num_centers)
        self.layer2 = RBFKANLayer(hidden_dim, output_dim, num_centers)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        return x