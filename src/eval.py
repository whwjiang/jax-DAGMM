import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import orbax.checkpoint as ocp

from tqdm import tqdm

from model import DAGMM
from utils import calc_sample_energies
from dataloader import cleanup_dataloader, get_dataloader


@nnx.jit
def eval_step(model: DAGMM, inputs: jnp.ndarray):
    """Evaluate for a single step."""
    gamma, _, z = model(inputs)
    _, mu, covariances = model.calc_mixture_stats(inputs, gamma, z)
    return gamma, z, mu, covariances


def eval(model: DAGMM, dataloader_train, dataloader_test):
    """Evaluate the model."""
    model.eval()
    
    N = 0
    mu_sum = 0
    cov_sum = 0
    gamma_sum = 0
    
    # first, collect global mixture parameters
    with tqdm(total=len(dataloader_train)) as pbar:
        pbar.set_description("Collecting mixture params")
        pbar.update(0)
        for step, (inputs, labels) in enumerate(dataloader_train):
            inputs = jax.tree.map(lambda x: jnp.array(x), inputs)
            gamma, _, mu, cov = eval_step(model, inputs)
            
            batch_gamma_sum = jnp.sum(gamma, axis=0)

            gamma_sum += batch_gamma_sum
            mu_sum += mu * jnp.expand_dims(batch_gamma_sum, -1)
            cov_sum += cov * jnp.expand_dims(jnp.expand_dims(batch_gamma_sum, -1), -1)
            
            N += inputs.shape[0]
            pbar.update(1)
            
    train_phi = gamma_sum / N
    train_mu = mu_sum / jnp.expand_dims(gamma_sum, -1)
    train_cov = cov_sum / jnp.expand_dims(jnp.expand_dims(gamma_sum, -1), -1)
            
    combined_z = []
    combined_energy = []
    combined_labels = []

    with tqdm(total=len(dataloader_train)) as pbar:
        pbar.set_description("Evaluating training data")
        pbar.update(0)
        for step, (inputs, labels) in enumerate(dataloader_train):
            inputs = jax.tree.map(lambda x: jnp.array(x), inputs)
            gamma, z, mu, cov = eval_step(model, inputs)
            energies = calc_sample_energies(train_phi, train_mu, train_cov, z)

            combined_z.append(z)
            combined_energy.append(energies)
            combined_labels.append(labels)
            pbar.update(1)

    # cleanup_dataloader(dataloader_train)

    with tqdm(total=len(dataloader_test)) as pbar:
        pbar.set_description("Evaluating test data")
        pbar.update(0)
        for step, (inputs, labels) in enumerate(dataloader_test):
            inputs = jax.tree.map(lambda x: jnp.array(x), inputs)
            gamma, z, mu, cov = eval_step(model, inputs)
            energies = calc_sample_energies(train_phi, train_mu, train_cov, z)

            combined_z.append(z)
            combined_energy.append(energies)
            combined_labels.append(labels)
            pbar.update(1)

    z = jnp.concatenate(combined_z, axis=0)
    energy = jnp.concatenate(combined_energy, axis=0)
    labels = np.concatenate(combined_labels, axis=0)

    cleanup_dataloader(dataloader_test)
    return z, energy, jnp.array(labels)


def load_checkpoint(model: DAGMM, dir="/tmp/checkpoints/dagmm"):
    model_state = nnx.state(model)
    with ocp.CheckpointManager(
        dir, options=ocp.CheckpointManagerOptions(read_only=True)
    ) as read_mgr:
        restored = read_mgr.restore(
            1,
            # pass in the model_state to restore the exact same State type
            args=ocp.args.Composite(state=ocp.args.PyTreeRestore(item=model_state)),
        )
    nnx.update(model, restored["state"])


def eval_from_checkpoint():
    key = jax.random.PRNGKey(42)
    batch_size = 1024
    key, dataloader_key = jax.random.split(key, 2)

    model = DAGMM(n_features=118, rngs=nnx.Rngs(key))
    dataloader_train = get_dataloader(dataloader_key, batch_size=batch_size, mode="train")
    dataloader_test = get_dataloader(None, batch_size=batch_size, mode="test")
    load_checkpoint(model)

    return eval(model, dataloader_train, dataloader_test)
