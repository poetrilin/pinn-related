# PINN's implementation for 2-D Possion equation

Solving 2-D Possion方程:$\nabla^2 u = f,$,Here we set

- $f(x,y)=-sin(x)sin(y),(x,y)\in [0,1]\times[0,1]$,
- boundary condition: $u(x,0)=u(x,\pi)=0,u(0,y)=u(\pi,y)=0$.

according to PINN's theorem , 

- $Loss_{res}=\|\nabla^2 u - f\|^2$,

- $Loss_{bc}=\|u(x,0)-0\|^2+\|u(x,\pi)-0\|^2+\|u(0,y)-0\|^2+\|u(\pi,y)-0\|^2$.

Here we simply let $L=Loss_{res}+Loss_{bc}$.




