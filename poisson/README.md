# PINN's implementation for 2-D Possion equation

Solving 2-D Possion Equations : $\nabla^2 u=f$ ,Here we set

- $f(x,y)=-sin(x)sin(y),(x,y)\in [0,1]\times[0,1]$,
- boundary condition: $u(x,0)=u(x,\pi)=0,u(0,y)=u(\pi,y)=0$.

according to PINN's theorem , 

- $Loss_{res}=\|\nabla^2 u - f\|^2$,

- $Loss_{bc}=\|u(x,0)-0\|^2+\|u(x,\pi)-0\|^2+\|u(0,y)-0\|^2+\|u(\pi,y)-0\|^2$.

Here we simply let $L=Loss_{res}+Loss_{bc}$ .


## Models 

1. PINN

parameters:
```bash
- n_hidden: 64
- n_hideen_layers: 3
- Optimizer: L-BFGS
- lr: 5e-2
- Epochs: 40
```
RESULT:
```bash
Relative Error: 1.7344e-02
rMAE: 1.9115e-02
rRMSE: 1.7344e-02
Number of parameters: 12737
```

2. FLS
   
(just let first activation function to be sin)

RESULT:

```bash
Epoch 40, Loss: 1.243544e-06
Relative Error: 9.7897e-03
rMAE: 9.6162e-03
rRMSE: 9.7897e-03
Number of parameters: 12737
```

3. Pinnformer
   
RESULT:

```bash
Relative Error: 9.4299e-02
rMAE: 1.0884e-01
rRMSE: 9.4299e-02
Number of parameters: 31609
```

??It seems that the Pinnformer is not well trained, maybe due to that the model is small?

4. kan

```bash
Relative Error: 3.3851e-02
rMAE: 2.8908e-02
rRMSE: 3.3851e-02
Number of parameters: 10560
# 但是训练时间久，约为3倍？
```

5. rbfkan
    
不好训,可能不收敛,alpha 如何确定？

```bash
Relative Error: 7.4790e-02
rMAE: 6.6253e-02
rRMSE: 7.4790e-02
Number of parameters: 8288
```

6. fftkan

```bash
Relative Error: 3.0648e-02
rMAE: 2.8069e-02
rRMSE: 3.0648e-02
Number of parameters: 11265
```

7. wavkan

```bash