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
| PINN    |  8577 | 1.524635e-06| 3.1991e-03 |3.5512e-03|170.79s|
| KAN-Mish    | 16720| 3.235757e-09 | 1.9934e-03 | 1.8858e-03| 1370.58s |
| KAN-SilU    | 16720|  4.482491e-09| 1.5055e-03|1.4048e-03|1251.76s
| PowerMLP-3-order| 2193 |3.851e-06| 4.6100e-03 | 5.4085e-03|329.17s|
| PowerMLP-4-order| 4353 |3.2919e-06| 1.8849e-02|2.2791e-02|519.71s|
| PowerMLP-tanh|

注:
Pinn: [2,64,64,64,1] 训练时间:170.79s

PowerMLP,很容易过拟合，宽度大一点，Loss降不下去但是rMAE和rRMSE 更差