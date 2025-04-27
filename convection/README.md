# Convection PDE

Solving 1-D convective equations : $\frac{\partial u}{\partial t}+\beta \frac{\partial u}{\partial x}=0$, where $x\in [0,2\pi],t\in [0,1]$ ,Here we set $\beta \in[1,0.1,0.001]$ in experiments.

- initial condition: $u(x,0)=sin(x)$,
- boundary condition: $u(0,t) = u(2\pi,t)$.

> The analytical solution to this PDE is $u(x, t) = sin(x −\beta t)$.

according to PINN's theorem :

- $Loss_{res}=||\frac{\partial u}{\partial t}+\beta \frac{\partial u}{\partial x}||^2$,
- $Loss_{ic}=||u(x,0)-sin(x)||^2$.
- $Loss_{bc}= ||u(0,t)-u(2\pi,t)||^2$.

Here we let $L=Loss_{res}+Loss_{bc}+Loss_{ic}$

# Results
- $\beta = 1$

| Models   | Paras  |  Loss         |  rMAE  | rRMSE |
| :----:   | :----:     | :----:       | :----: | :----: |
| pinn     | 21905    |   3.149170e-06 |3.3054e-01 | 4.1534e-01|
| kan-silu | 11660    | 6.877966e-07|||
| kan-mish | 11660    |  6.531657e-07|||
| powermlp-silu | 23889| 2.470467e-06 |||
| powermlp-mish | 23889|   3.028087e-06|||

 
- $\beta = 0.1$

| Models   | Paras  |  Loss         |  rMAE  | rRMSE |
| :----:   | :----:     | :----:       | :----: | :----: |
| pinn     | 21905    | 1.660654e-06 |4.4622e-02 |6.0658e-02 |
| kan-silu | 11660    |  8.058688e-09  |||
| kan-mish | 11660    |   4.943538e-09 |||
| powermlp-silu | 23889 | 1.813292e-06 |||
| powermlp-mish | 23889|   2.375815e-06 |||

- $\beta = 0.01$

| Models   | Paras  |  Loss         |  rMAE  | rRMSE |
| :----:   | :----:     | :----:       | :----: | :----: |
| pinn     | 21905   | 7.984570e-07 | 4.2186e-02|6.0658e-02|
| kan-silu | 11660   |   7.731960e-09  |||
| kan-mish | 11660    |  6.362592e-09 |||
| powermlp-silu | 23889 |  1.494560e-06 |||
| powermlp-mish | 23889|  1.031550e-06|||
