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

Here, we are specifying β = 3.The equation has a simple analytical solution:
$$
u(x, t) = \sin(\pi x) cos(2\pi t) + \frac{1}{2}\sin(\beta \pi x) \cos(2\beta \pi t)
$$


- $Loss_{res} = || u_{tt}-\beta u_{xx}||^2$
- $Loss_{ic} = ||u(x,0)-\sin(\pi x) - \frac{1}{2}\sin(\beta \pi x)||^2 + ||u_t(x,0)||^2$
- $Loss_{bc} = ||u(0,t)||^2 + ||u(1,t)||^2$ 