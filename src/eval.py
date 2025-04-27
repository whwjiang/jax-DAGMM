import jax
import jax.numpy as jnp
from flax import nnx
from tqdm import tqdm

from model import DAGMM
from utils import calc_mixture_stats, calc_sample_energies
from dataloader import cleanup_dataloader

@nnx.jit
def eval_step(model: DAGMM, inputs: jnp.ndarray):
    """Evaluate for a single step."""
    gamma, _, z = model(inputs)
    phi, mu, covariances = calc_mixture_stats(inputs, gamma, z)
    energies = calc_sample_energies(model.k, z, phi, mu, covariances)
    return z, energies

def eval(model: DAGMM, dataloader_train, dataloader_test):
    """Evaluate the model."""
    model.eval()

    combined_z = jax.device_put([])
    combined_energy = jax.device_put([])
    combined_labels = []

    with tqdm(total=len(dataloader_train)) as pbar:
        pbar.set_description("1. Evaluating training data")
        pbar.update(0)
        for step, (inputs, labels) in enumerate(dataloader_train):
            inputs = jax.tree.map(lambda x: jnp.array(x), inputs)
            z, energies = eval_step(model, inputs)
            
            combined_z.append(z)
            combined_energy.append(energies)
            combined_labels.append(labels)
            pbar.update(1)
            
    cleanup_dataloader(dataloader_train)
    
    with tqdm(total=len(dataloader_test)) as pbar:
        pbar.set_description("2. Evaluating test data")
        pbar.update(0)
        for step, (inputs, labels) in enumerate(dataloader_test):
            inputs = jax.tree.map(lambda x: jnp.array(x), inputs)
            z, energies = eval_step(model, inputs)
            
            combined_z.append(z)
            combined_energy.append(energies)
            combined_labels.append(labels)
            pbar.update(1)

    z = jnp.concatenate(combined_z, axis=0)
    energy = jnp.concatenate(combined_energy, axis=0)
    labels = jnp.concatenate(combined_labels, axis=0)
    
    cleanup_dataloader(dataloader_test)
    return z, energy, labels