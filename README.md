# AI4pde implementation 



> Reference environment:
> 
> torch == 2.1.0
> 
> python == 3.10.15

## Models 


## PDEs

### 2-D Possion equation ✅

Solving 2-D Possion Equations : $\nabla^2 u=f$ ,Here we set

- $f(x,y)=-sin(x)sin(y),(x,y)\in [0,1]\times[0,1]$,

specific inplementation in [poisson](poisson\README.md)

### 1D Burger's equation :x:

Data Driven: We use DeepONet and FNO

1D Burgers' equation: 
$$
\frac{\partial u} {\partial t}+u \frac{\partial u} {\partial x}=\nu\frac{\partial^{2} u} {\partial x^{2}} 
$$

using $t \in [0,2)$ infer $u(x,t=2)$