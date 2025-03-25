import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        super(SpectralConv1d, self).__init__() 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.scale = (1 / math.sqrt(in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
    @staticmethod
    def compl_mul1d(input, weights):
        # Complex multiplication
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)
    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x) # real 傅立叶变换，dim

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat) 
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    
class FNOBlock1d(nn.Module):
    def __init__(self, modes, width, activation = F.relu):
        super(FNOBlock1d, self).__init__()
        self.modes1 = modes
        self.width = width
        self.w = nn.Conv1d(self.width, self.width, kernel_size=1)
        self.conv = SpectralConv1d(self.width, self.width, self.modes1)
        self.act = activation

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if self.act is not None:
            x = self.act(x)
        return x
class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes1 = modes
        self.width = width
         ## 提升维度部分
        self.fc0 = nn.Linear(1, self.width) # input channel is 2: (a(x), x)
         ## 傅立叶层部分
        self.block1 = FNOBlock1d(modes, width)
        self.block2 = FNOBlock1d(modes, width)
        self.block3 = FNOBlock1d(modes, width)
        self.block4 = FNOBlock1d(modes, width, activation=None)
 
        ## 降低维度部分
        self.fc1 = nn.Linear(self.width, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward(self, x):
        # batchsize = x.shape[0]
        # size = x.shape[1]
        
        x = self.fc0(x)
        # x = x.permute(0, 2, 1)
        
        x= self.block1(x)
        x= self.block2(x) 
        x= self.block3(x)
        x= self.block4(x)

        # x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)