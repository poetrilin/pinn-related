# https://jaxkan.readthedocs.io/en/latest/tutorials/Tutorial%204%20-%20Adaptive%20State%20Transition.html
from jaxkan.KAN import KAN
from jaxkan.utils.general import adam_transition

import jax
import jax.numpy as jnp

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from flax import nnx
import optax

import matplotlib.pyplot as plt
import numpy as np

seed = 42
def f(x1,x2,x3,x4):
    return jnp.exp(0.5*jnp.sin((jnp.pi * x1**2) + (jnp.pi * x2**2)) + 0.5*jnp.sin((jnp.pi * x3**2) + (jnp.pi * x4**2)))

def generate_data(minval=-1, maxval=1, num_samples=1000, seed=42):
    key = jax.random.PRNGKey(seed)
    x1_key, x2_key, x3_key, x4_key = jax.random.split(key, 4)

    x1 = jax.random.uniform(x1_key, shape=(num_samples,), minval=minval, maxval=maxval)
    x2 = jax.random.uniform(x2_key, shape=(num_samples,), minval=minval, maxval=maxval)
    x3 = jax.random.uniform(x3_key, shape=(num_samples,), minval=minval, maxval=maxval)
    x4 = jax.random.uniform(x4_key, shape=(num_samples,), minval=minval, maxval=maxval)

    y = f(x1, x2, x3, x4).reshape(-1, 1)
    X = jnp.stack([x1, x2, x3, x4], axis=1)

    return X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# Initialize a KAN model
n_in = X_train.shape[1]
n_out = y_train.shape[1]
n_hidden = 10

layer_dims = [n_in, n_hidden, n_out]

req_params = {'k': 3, 'G': 3, 'grid_e': 0.02}

model = KAN(layer_dims = layer_dims,
            layer_type = 'base',
            required_parameters = req_params,
            seed = seed
           )