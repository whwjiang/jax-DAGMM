import jax.numpy as jnp

from sklearn.metrics import precision_score, recall_score, f1_score
from jax.scipy.linalg import inv, cholesky



def cosine_similarity(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    norm_x = jnp.linalg.norm(x, axis=1, keepdims=True)
    norm_y = jnp.linalg.norm(y, axis=1, keepdims=True)
    dot_product = jnp.sum(norm_x * norm_y, axis=1)
    return dot_product


def euclidean_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(x - y, ord=2, axis=1) / jnp.linalg.norm(x, ord=2, axis=1)


def calc_mixture_stats(inputs, gamma, z):
    # gamma.shape: (1024, 4); x_hat.shape: (1024, 122); z.shape: (1024, 3)
    # phi: same shape as gamma
    gamma_sum = jnp.sum(gamma, axis=0)
    phi = gamma_sum / inputs.shape[0]
    # mu: (4, 3)
    gamma_expanded = jnp.expand_dims(gamma, -1)  # Shape (1024, 4, 1)
    z_expanded = jnp.expand_dims(z, 1)  # Shape (1024, 1, 3)
    mu = jnp.sum(gamma_expanded * z_expanded, axis=0)
    mu /= jnp.expand_dims(gamma_sum, -1)
    # covariances: (4, 3, 3)
    z_mu = z_expanded - jnp.expand_dims(mu, 0)  # Shape (1024, 4, 3)
    z_mu_z_mu_t = jnp.expand_dims(z_mu, -1) * jnp.expand_dims(z_mu, -2)
    covariances = jnp.sum(jnp.expand_dims(gamma_expanded, -1) * z_mu_z_mu_t, axis=0)
    covariances /= jnp.expand_dims(jnp.expand_dims(gamma_sum, -1), -1)
    return phi, mu, covariances


def calc_sample_energies(k, z, phi, mu, covs):
    # broadcast differences
    z_mu = jnp.expand_dims(z, 1) - jnp.expand_dims(mu, 0)  # Shape: (N, K, D)

    eps = 1e-12
    cov_invs = []
    cov_dets = []

    # Loop over components and calculate the covariance inverse and determinant
    for i in range(k):
        cov_i = covs[i] + (jnp.eye(covs[i].shape[-1]) * eps)  # numerical stability
        cov_inv_i = inv(cov_i)
        cov_invs.append(jnp.expand_dims(cov_inv_i, 0))  # Shape: (1, D, D)

        L = cholesky(cov_i * (2 * jnp.pi), lower=True)
        cov_det_i = jnp.prod(jnp.diagonal(L)) ** 2
        cov_dets.append(jnp.expand_dims(cov_det_i, 0))  # Shape: (1,)

    cov_invs = jnp.concatenate(cov_invs, axis=0)  # Shape: (K, D, D)
    cov_dets = jnp.expand_dims(jnp.sqrt(jnp.concatenate(cov_dets)), 0)  # Shape: (K, 1)

    E_z = jnp.sum(jnp.expand_dims(z_mu, -1) * jnp.expand_dims(cov_invs, 0), axis=-2)
    E_z = jnp.sum(E_z * z_mu, axis=-1)
    E_z = jnp.exp(-0.5 * E_z)
    E_z = -jnp.log(jnp.sum(jnp.expand_dims(phi, 0) * E_z / cov_dets, axis=1) + eps)
    return E_z


# def load_checkpoint(model: DAGMM, dir="/tmp/checkpoints/dagmm"):
#     model_state = nnx.state(model)
#     with ocp.CheckpointManager(
#         dir, options=ocp.CheckpointManagerOptions(read_only=True)
#     ) as read_mgr:
#         restored = read_mgr.restore(
#             1,
#             # pass in the model_state to restore the exact same State type
#             args=ocp.args.Composite(state=ocp.args.PyTreeRestore(item=model_state)),
#         )
#     nnx.update(model, restored["state"])
#     return model
