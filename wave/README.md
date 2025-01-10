# 1D-Wave Equation

The 1D-Wave equation is a hyperbolic PDE that is used to describe the propagation of waves in one spatial dimension. It is often used in physics and engineering to model various wave phenomena, such as sound waves, seismic waves, and electromagnetic waves. The system has the formulation with periodic boundary conditions as follows:


$$
\frac{\partial^{2} u} {\partial t^{2}}-\beta\frac{\partial^{2} u} {\partial x^{2}}=0, \; \forall x \in[ 0, 1 ], \; t \in[ 0, 1 ] 
$$
$$
\begin{aligned} {{\mathrm{IC} :}} & {{} {u(x,0)=\sin ( \pi x )+\frac{1} {2} \sin( \beta\pi x ), \ \ {\frac{\partial u ( x, 0 )} {\partial t}}=0}}  \\
\mathrm{BC}: &  u ( 0, t )= u ( 1, t )=0
\end{aligned} 
$$

Here, we are specifying β = 1.The equation has a simple analytical solution:
$$
u(x, t) = \sin(\pi x) \cos(\pi \sqrt{\beta} t) + \frac{1}{2} \sin(\beta \pi x) \cos(\beta \pi \sqrt{\beta} t).
$$


- $Loss_{res} = || u_{tt}-\beta u_{xx}||^2$
- $Loss_{ic} = ||u(x,0)-\sin(\pi x) - \frac{1}{2}\sin(\beta \pi x)||^2 + ||u_t(x,0)||^2$
- $Loss_{bc} = ||u(0,t)||^2 + ||u(1,t)||^2$ 

Loss = $Loss_{res} + 0.5 Loss_{ic} + 0.4 Loss_{bc}$
# Results

- $\beta=1$

| Models  | Paras | Loss | rMAE | rRMSE | 
| :----:  | :----: | :----: | :----: | :----: | 
|  Pinn   |24993| 7.706309e-06| 2.2889e-03 | 2.2889e-03|
|kan-SiLU|11660|4.309552e-05| 1.5055e-03|1.4048e-03|
|kan-Mish|11660|3.963015e-05| 2.9726e-03|3.0355e-03|
|powermlp|16737|5.155272e-05|2.3757e-03|2.3717e-03|

- $\beta=4$

| Models  | Paras | Loss | rMAE | rRMSE | 
| :----:  | :----: | :----: | :----: | :----: | 
| kan|11660 |  3.895181e-02 |
