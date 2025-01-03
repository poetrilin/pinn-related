import math
import matplotlib.pyplot as plt
import numpy as np
# 可添加

def burgers_real(X, T, nu):
    # 输入: 
    # X: 区间[-1,1]上的点构成的Nx维数组
    # T: 区间[0,1]上的点构成的Nt维数组
    # nu: Burgers方程中的参数ν
    #
    # 输出: 
    # U: 一个Nt * Nx维数组, U[j,i]表示在点(T[j],X[i])处Burgers方程的解.
    
    Nx = len(X)
    Nt = len(T)
    U = np.zeros([Nt,Nx]) 
    # TODO

nu = 0.01/math.pi

X, T = np.meshgrid(np.linspace(-1,1,201), np.linspace(0,1,101)) 

U = burgers_real(X[0,:],T[:,0],nu)

lower_bound = np.min(U)
upper_bound = np.max(U)
print(lower_bound, upper_bound)
fig, ax = plt.subplots()
plt.title("Data")
levels = np.linspace(lower_bound,upper_bound,100) #对颜色渐进细致程度进行设置
cs = ax.contourf(T, X, U, levels,cmap=plt.get_cmap('Spectral'))
cbar = fig.colorbar(cs) #添加colorbar
plt.savefig('fig.eps')  #保存图片为eps格式
plt.savefig('fig.png')  #保存图片为png格式