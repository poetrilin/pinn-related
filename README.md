# AI4pde implementation 

## Models 


## PDEs

### 2-D Possion equation

Solving 2-D Possion Equations : $\nabla^2 u=f$ ,Here we set

- $f(x,y)=-sin(x)sin(y),(x,y)\in [0,1]\times[0,1]$,

### 1D Burger's equation

Data Driven: We use DeepONet and FNO

1D Burgers' equation: 
$$
\frac{\partial u} {\partial t}+u \frac{\partial u} {\partial x}=\nu\frac{\partial^{2} u} {\partial x^{2}} 
$$

using $t \in [0,2)$ infer $u(x,t=2)$