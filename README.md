# PINN's implementation for 2-D Possion equation

Solving 2-D Possion Equations : $\nabla^2 u =f,$,Here we set

- $f(x,y)=-sin(x)sin(y),(x,y)\in [0,1]\times[0,1]$,
- boundary condition: $u(x,0)=u(x,\pi)=0,u(0,y)=u(\pi,y)=0$.

according to PINN's theorem , 

- $Loss_{res}=\|\nabla^2 u - f\|^2$,

- $Loss_{bc}=\|u(x,0)-0\|^2+\|u(x,\pi)-0\|^2+\|u(0,y)-0\|^2+\|u(\pi,y)-0\|^2$.

Here we simply let $L=Loss_{res}+Loss_{bc}$.

> reference environment:
> torch == 2.1.0
> python == 3.10.15
> 
## Models 
1. PINN

parameters:
```bash
- n_hidden: 64
- n_hideen_layers: 3
- Optimizer: L-BFGS
- lr: 1e-3
- Epochs: 30
```
RESULT:
```bash
Epoch 30, Loss: 1.541266e-06
Relative Error: 1.3132e-02
rMAE: 1.4058e-02
rRMSE: 1.2832e-02
```

2. FLS
(just let first activation function to be sin)
RESULT:
```bash
Epoch 30, Loss: 1.243544e-06
Relative Error: 6.3557e-03
rMAE: 6.5656e-03
rRMSE: 6.2189e-03
```

3. Pinnformer
4. kan




