from typing import Literal

ACT = Literal["relu", "tanh","sigmoid", "silu","gelu","leakyrelu"] 
MODELS = Literal["pinn","pinnsformer","kan","fls","rbfkan","fftkan","wavkan"]