# PINN's implementation for 2-D Possion equation

Solving 2-D Possion Equations : $\nabla^2 u=f$ ,Here we set

- $f(x,y)=-sin(x)sin(y),(x,y)\in [0,1]\times[0,1]$,
- boundary condition: $u(x,0)=u(x,\pi)=0,u(0,y)=u(\pi,y)=0$.

according to PINN's theorem , 

- $Loss_{res}=\|\nabla^2 u - f\|^2$,

- $Loss_{bc}=\|u(x,0)-0\|^2+\|u(x,\pi)-0\|^2+\|u(0,y)-0\|^2+\|u(\pi,y)-0\|^2$.

Here we simply let $L=Loss_{res}+Loss_{bc}$ .


## Result

| Models  | Paras | Loss | rMAE | rRMSE | Training Time |
| :----:  | :----: | :----: | :----: | :----: | :----: |
| PINN    |  12737 | 7.049e-07| 3.324e-03 |3.282e-03|170.79s|
| KAN     | 16720| 1.164349e-08 | 1.6987e-03 | 1.6362e-03| 1903.70s |
| PowerMLP-3-order| 2193 |3.851e-06| 4.6100e-03 | 5.4085e-03|329.17s|
| PowerMLP-4-order| 4353 |3.2919e-06| 1.8849e-02|2.2791e-02|519.71s|


PowerMLP,很容易过拟合，宽度大一点，Loss降不下去但是rMAE和rRMSE 更差