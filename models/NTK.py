"""
refer to https://pytorch.ac.cn/tutorials/intermediate/neural_tangent_kernels.html
"""
import torch
import torch.nn as nn
from torch.func import functional_call, vmap, vjp, jvp, jacrev
device = 'cuda' if torch.cuda.device_count() > 0 else 'cpu'


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 32, (3, 3))
        self.fc = nn.Linear(21632, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.relu()
        x = self.conv2(x)
        x = x.relu()
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
x_train = torch.randn(20, 3, 32, 32, device=device)
x_test = torch.randn(5, 3, 32, 32, device=device)

net = CNN().to(device)

# Detaching the parameters because we won't be calling Tensor.backward().
params = {k: v.detach() for k, v in net.named_parameters()}

def fnet_single(params, x):
    return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

result = empirical_ntk_jacobian_contraction(fnet_single, params, x_train, x_test)
print(result.shape)