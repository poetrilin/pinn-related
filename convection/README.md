
Solving 1-D convective equations : $\frac{\partial u}{\partial t}+\beta \frac{\partial u}{\partial x}=0$, where $x\in [0,2\pi],t\in [0,1]$ ,Here we set $\beta =40$ in experiments.

- initial condition: $u(x,0)=sin(x)$,
- boundary condition: $u(0,t) = u(2\pi,t)$.

> The analytical solution to this PDE is $u(x, t) = sin(x −\beta t)$.


according to PINN's theorem , 

- $Loss_{res}=||\frac{\partial u}{\partial t}+\beta \frac{\partial u}{\partial x}||^2$,

- $Loss_{bc}= ||u(0,t)-u(2\pi,t)||^2$.

Here we simply let $L=Loss_{res}+Loss_{bc}$ .
