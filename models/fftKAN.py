import torch 
import torch.nn as nn
import numpy as np

# This is inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients
# It should be easier to optimize as fourier are more dense than spline (global vs local)
# Once convergence is reached you can replace the 1d function with spline approximation for faster evaluation giving almost the same result
# The other advantage of using fourier over spline is that the function are periodic, and therefore more numerically bounded
# Avoiding the issues of going out of grid

class NaiveFourierKANLayer(nn.Module):
    def __init__( self, inputdim, outdim, gridsize, addbias=True, smooth_initialization=True):
        super(NaiveFourierKANLayer,self).__init__()
        self.gridsize= gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        
        # With smooth_initialization, fourier coefficients are attenuated by the square of their frequency.
        # This makes KAN's scalar functions smooth at initialization.
        # Without smooth_initialization, high gridsizes will lead to high-frequency scalar functions,
        # with high derivatives and low correlation between similar inputs.
        grid_norm_factor = (torch.arange(gridsize) + 1)**2 if smooth_initialization else np.sqrt(gridsize)
        
        #The normalization has been chosen so that if given inputs where each coordinate is of unit variance,
        #then each coordinates of the output is of unit variance 
        #independently of the various sizes
        self.fouriercoeffs = torch.nn.Parameter( torch.randn(2,outdim,inputdim,gridsize) / 
                                                (np.sqrt(inputdim) * grid_norm_factor ) )
        if( self.addbias ):
            self.bias  = torch.nn.Parameter( torch.zeros(1,outdim))

    #x.shape ( ... , indim ) 
    #out.shape ( ..., outdim)
    def forward(self,x):
        xshp = x.shape
        outshape = xshp[0:-1]+(self.outdim,)
        x = torch.reshape(x,(-1,self.inputdim))
        #Starting at 1 because constant terms are in the bias
        k = torch.reshape( torch.arange(1,self.gridsize+1,device=x.device),(1,1,1,self.gridsize))
        xrshp = torch.reshape(x,(x.shape[0],1,x.shape[1],1) ) 
        #This should be fused to avoid materializing memory
        c = torch.cos( k*xrshp ) 
        s = torch.sin( k*xrshp )
        #We compute the interpolation of the various functions defined by their fourier coefficient for each input coordinates and we sum them 
        y =  torch.sum( c*self.fouriercoeffs[0:1],(-2,-1)) 
        y += torch.sum( s*self.fouriercoeffs[1:2],(-2,-1))
        if( self.addbias):
            y += self.bias
        #End fuse
        '''
        #You can use einsum instead to reduce memory usage
        #It stills not as good as fully fused but it should help
        #einsum is usually slower though
        c = th.reshape(c,(1,x.shape[0],x.shape[1],self.gridsize))
        s = th.reshape(s,(1,x.shape[0],x.shape[1],self.gridsize))
        y2 = th.einsum( "dbik,djik->bj", th.concat([c,s],axis=0) ,self.fouriercoeffs )
        if( self.addbias):
            y2 += self.bias
        diff = th.sum((y2-y)**2)
        print("diff")
        print(diff) #should be ~0
        '''
        y = torch.reshape( y, outshape)
        return y

class fftKAN(nn.Module):
    def __init__(self, inputdim, outdim, hidden_dim=None, gridsize=5, hidden_layers=1, addbias=True, smooth_initialization=True):
        super(fftKAN,self).__init__()
        
        self.fkan_in = NaiveFourierKANLayer(inputdim, hidden_dim, gridsize, addbias, smooth_initialization)
        self.hiddens = nn.ModuleList([NaiveFourierKANLayer(hidden_dim, hidden_dim, gridsize, addbias, smooth_initialization) for _ in range(hidden_layers)])
        self.fkan_out = NaiveFourierKANLayer(hidden_dim, outdim, gridsize, addbias, smooth_initialization)
        self.act = torch.tanh
    def forward(self,x):
        x = self.fkan_in(x)
        x = self.act(x)

        for layer in self.hiddens:
            x = layer(x)
            x = self.act(x)
        x = self.fkan_out(x)
        return x

# def demo():
#     bs = 10
#     L = 3 #Not necessary just to show that additional dimensions are batched like Linear
#     inputdim = 50
#     hidden = 200
#     outdim = 100
#     gridsize = 300

#     device = "cpu" #"cuda"

#     fkan1 = NaiveFourierKANLayer(inputdim, hidden, gridsize).to(device)
#     fkan2 = NaiveFourierKANLayer(hidden, outdim, gridsize).to(device)

#     x0 =torch.randn(bs,inputdim).to(device)

#     h = fkan1(x0)
#     y = fkan2(h)
#     print("x0.shape")
#     print( x0.shape)
#     print("h.shape")
#     print( h.shape)
#     print( "torch.mean( h )")
#     print( torch.mean( h ) )
#     print( "torch.mean( torch.var(h,-1) )")
#     print( torch.mean( torch.var(h,-1)))

#     print("y.shape")
#     print( y.shape )
#     print( "torch.mean( y)")
#     print( torch.mean( y ) )
#     print( "torch.mean( torch.var(y,-1) )")
#     print( torch.mean( torch.var(y,-1)))

#     print(" ")
#     print(" ")
#     print("Sequence example")
#     print(" ")
#     print(" ")
#     xseq =torch.randn(bs, L ,inputdim).to(device)

#     h = fkan1(xseq)
#     y = fkan2(h)
#     print("xseq.shape")
#     print( xseq.shape)
#     print("h.shape")
#     print( h.shape)
#     print( "torch.mean( h )")
#     print( torch.mean( h ) )
#     print( "torch.mean( torch.var(h,-1) )")
#     print( torch.mean( torch.var(h,-1)))

#     print("y.shape")
#     print( y.shape )
#     print( "torch.mean( y)")
#     print( torch.mean( y ) )
#     print( "torch.mean( torch.var(y,-1) )")
#     print( torch.mean( torch.var(y,-1)))

# if __name__ == "__main__":
#     demo()