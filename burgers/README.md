## Burgers' Equation

$$
\frac{\partial u} {\partial t}+u \frac{\partial u} {\partial x}=\nu\frac{\partial^{2} u} {\partial x^{2}} 
$$

where $x\in [0,2 \pi ],t\in [0,2]$ ,Here we set $\nu =0.01/ \pi$ in experiments.

- $u(x,0)=−\sin (\pi x)$ ,
- $u(−1,t)=u(1,t)=0$,
​
> The true solution is 

- $ Loss_{res} =  \| \frac{\partial u} {\partial t}+u \frac{\partial u} {\partial x}-\nu\frac{\partial^{2} u} {\partial x^{2}} \|^2$
- $ Loss_{bc} =  \| u(−1,t) \|^2 + \| u(1,t) \|^2$
- $ Loss_{ic} =  \| u(x,0)+\sin (\pi x) \|^2$