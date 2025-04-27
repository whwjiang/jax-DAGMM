import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from IPython.display import clear_output

from model import DAGMM
from utils import calc_mixture_stats, calc_sample_energies


def _objective_fn(model: DAGMM, inputs):
    gamma, x_hat, z = model(inputs)
    
    n = inputs.shape[0]
    phi, mu, covariances = calc_mixture_stats(inputs, gamma, z)
    mse = jnp.mean((inputs - x_hat) ** 2)
    energy = jnp.mean(calc_sample_energies(model.k, z, phi, mu, covariances))
    reg_1 = (model.lambda_1 / n) * energy
    reg_2 = model.lambda_2 * (jnp.sum(jnp.diagonal(covariances, axis1=1, axis2=2)) ** -1)
    
    return mse + reg_1 + reg_2

@nnx.jit
def _train_step(model: DAGMM, optimizer: nnx.Optimizer, 
               metrics: nnx.MultiMetric, inputs: jnp.ndarray):
    """Train for a single step."""
    grad_fn = nnx.value_and_grad(_objective_fn)
    loss, grads = grad_fn(model, inputs)
    metrics.update(loss=loss)
    optimizer.update(grads)
    
def _train_epoch(model: DAGMM, optimizer: nnx.Optimizer,
                metrics: nnx.MultiMetric, dataloader: DataLoader):
    """Train for a single epoch."""
    for step, (inputs, _) in enumerate(dataloader):
        inputs = jax.tree.map(lambda x: jnp.array(x), inputs)
        _train_step(model, optimizer, metrics, inputs)
        
def save_checkpoint(model: DAGMM, dir="/tmp/checkpoints/dagmm"):
    checkpoint_manager = ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty(dir),
        options=ocp.CheckpointManagerOptions(
            max_to_keep=2,
            keep_checkpoints_without_metrics=False,
            create=True,
        ),
    )

    model_state = nnx.state(model)
    checkpoint_manager.save(
        1, args=ocp.args.Composite(state=ocp.args.PyTreeSave(model_state))
    )
    checkpoint_manager.close()

def train(model: DAGMM, optimizer: nnx.Optimizer, 
          metrics: nnx.MultiMetric, dataloader: DataLoader, 
          epochs: int, save_model: bool = True):
    """Train the model."""
    metrics_history = {
        'train_loss': [],
    }
    model.train()

    for epoch in range(epochs):
        _train_epoch(model, optimizer, metrics, dataloader)
        
        for metric, value in metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)
        metrics.reset() 

        clear_output(wait=True)
        # Plot loss and accuracy in subplots
        fig, (ax1) = plt.subplots(1, 1, figsize=(15, 5))
        ax1.set_title('Loss')
        ax1.plot(metrics_history['train_loss'], label=f'train_loss')
        ax1.legend()
        plt.show()
        
    if save_model:
        save_checkpoint(model)
        print("Model saved.")