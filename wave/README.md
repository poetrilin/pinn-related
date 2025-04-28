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

The equation has a simple analytical solution:
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
|  Pinn       |24993| 7.706309e-06| 2.2889e-03 | 2.2889e-03|
|kan-SiLU     |11660|4.309552e-05  | 2.4503e-03|2.4503e-03|
|powermlp-SiLU|16737|5.155272e-05|2.3757e-03|2.3717e-03|
|kan-Mish     |11660|3.963015e-05  | 2.9726e-03|3.0355e-03|
|powermlp-Mish|16737|5.155272e-05|1.6169e-03|1.6247e-03|

- $\beta=0.1$

| Models   | Paras      |  Loss         |  rMAE  | rRMSE | 
| :----:   | :----:     |  :----:       | :----: | :----: | 
| pinn     | 24993      |  1.447567e-04 | 4.2955e-02| 8.8862e-02|
| kan-silu | 11660      |  7.227323e-05 | 4.1698e-02| 8.9851e-02|
| powermlp | 16737      |  2.019787e-04 | 4.3012e-02| 8.7547e-02|
| kan-mish | 11660      |  7.432827e-05 | 4.1744e-02| 8.9733e-02|
| powermlp-mish | 16737 |  3.529122e-04 | 4.3001e-02| 8.2085e-02|

- $\beta = 0.01 $

| Models   | Paras  |  Loss         |  rMAE  | rRMSE | Training Time |
| :----:   | :----:     | :----:       | :----: | :----: | :----: |
| pinn     | 21905      | 3.119743e-07 | 2.0152e-03 | 4.5697e-03  |   |
| kan-silu | 11660      | 6.877966e-07 | 1.5569e-03 | 5.1216e-03  |   |
| powermlp-silu | 16737 | 2.497201e-06 | 1.7187e-03 | 4.5350e-03  | |
| kan-mish | 11660      | 6.531657e-07| 1.5265e-03 | 5.1051e-03   | |
| powermlp-mish | 16737 | 2.352901e-06 |  1.6377e-03  |  4.5209e-03  ||