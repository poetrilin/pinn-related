from jaxkan.KAN import KAN

import jax
import jax.numpy as jnp
from jaxkan.utils.PIKAN import sobol_sample, gradf
from flax import nnx
import optax
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

seed = 42

# Generate Collocation points for PDE
N = 2**12
collocs = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,1]), N, seed)) # (4096, 2)

# Generate Collocation points for BCs
N = 2**6

BC1_colloc = jnp.array(sobol_sample(np.array([0,-1]), np.array([0,1]), N)) # (64, 2)
BC1_data = - jnp.sin(np.pi*BC1_colloc[:,1]).reshape(-1,1) # (64, 1)

BC2_colloc = jnp.array(sobol_sample(np.array([0,-1]), np.array([1,-1]), N)) # (64, 2)
BC2_data = jnp.zeros(BC2_colloc.shape[0]).reshape(-1,1) # (64, 1)

BC3_colloc = jnp.array(sobol_sample(np.array([0,1]), np.array([1,1]), N)) # (64, 2)
BC3_data = jnp.zeros(BC3_colloc.shape[0]).reshape(-1,1) # (64, 1)

# Create lists for BCs
bc_collocs = [BC1_colloc, BC2_colloc, BC3_colloc]
bc_data = [BC1_data, BC2_data, BC3_data]


# Initialize a KAN model
n_in = collocs.shape[1]
n_out = 1
n_hidden = 6

layer_dims = [n_in, n_hidden, n_hidden, n_out]
req_params = {'D': 5, 'flavor':'modified', 'external_weights':True}

model = KAN(layer_dims = layer_dims,
            layer_type = 'chebyshev',
            required_parameters = req_params,
            seed = seed
           )

print(model)

opt_type = optax.adam(learning_rate=0.001)

optimizer = nnx.Optimizer(model, opt_type)

# PDE Loss
def pde_loss(model, collocs):
    # Eq. parameter
    v = jnp.array(0.01/jnp.pi, dtype=float)

    def u(x):
        y = model(x)
        return y

    # Physics Loss Terms
    u_t = gradf(u, 0, 1)
    u_x = gradf(u, 1, 1)
    u_xx = gradf(u, 1, 2)

    # Residual
    pde_res = u_t(collocs) + model(collocs)*u_x(collocs) - v*u_xx(collocs)

    return pde_res

# Define train loop
@nnx.jit
def train_step(model, optimizer, collocs, bc_collocs, bc_data):

    def loss_fn(model):
        pde_res = pde_loss(model, collocs)
        total_loss = jnp.mean((pde_res)**2)

        # Boundary losses
        for idx, colloc in enumerate(bc_collocs):
            # Residual = Model's prediction - Ground Truth
            residual = model(colloc)
            residual -= bc_data[idx]
            # Loss
            total_loss += jnp.mean(residual**2)

        return total_loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)

    return loss

# Initialize train_losses
num_epochs = 5000
train_losses = jnp.zeros((num_epochs,))

for epoch in tqdm(range(num_epochs), desc='Training',total=num_epochs):
    # Calculate the loss
    loss = train_step(model, optimizer, collocs, bc_collocs, bc_data)

    # Append the loss
    train_losses = train_losses.at[epoch].set(loss)


plt.figure(figsize=(7, 4))

plt.plot(np.array(train_losses), label='Train Loss', marker='o', color='#25599c', markersize=1)

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.yscale('log')

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.show()

def metric(N_t=100, N_x=256,save_path =None):
    # Generate test points
    t = np.linspace(0.0, 1.0, N_t)
    x = np.linspace(-1.0, 1.0, N_x)
    T, X = np.meshgrid(t, x, indexing='ij')
    coords = np.stack([T.flatten(), X.flatten()], axis=1)
    # Get the model output
    output = model(jnp.array(coords))
    resplot = np.array(output).reshape(N_t, N_x)
    plt.figure(figsize=(7, 4))
    plt.pcolormesh(T, X, resplot, shading='auto', cmap='Spectral_r')
    plt.colorbar()

    plt.title('Solution of Burgers Equation')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.tight_layout()
    if isinstance(save_path,str):
        plt.savefig(save_path)
    else:
        plt.show() 

metric()

 

