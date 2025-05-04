import jax
import jax.numpy as jnp
from sklearn.metrics import precision_recall_fscore_support as prf
from jax.scipy.linalg import solve_triangular
from jax.scipy.special import logsumexp


def cosine_similarity(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    norm_x = jnp.linalg.norm(x, ord=2, axis=1)
    norm_y = jnp.linalg.norm(y, ord=2, axis=1)
    dot_product = jnp.sum(x * y, axis=1)
    return dot_product / (norm_x * norm_y)


def euclidean_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(x - y, ord=2, axis=1) / jnp.linalg.norm(x, ord=2, axis=1)


def calc_sample_energies(phi: jnp.ndarray,
                         mu:  jnp.ndarray,
                         covs: jnp.ndarray,
                         z:   jnp.ndarray,
                         eps: float = 1e-12) -> jnp.ndarray:

    N, D = z.shape

    covs_j = covs + eps * jnp.eye(D)
    L = jnp.linalg.cholesky(covs_j)           # → (K, D, D)

    diag_L       = jnp.diagonal(L, axis1=-2, axis2=-1)  # (K, D)
    log_det_Sig  = 2.0 * jnp.sum(jnp.log(diag_L), axis=1)  # (K,)

    const = -0.5 * (D * jnp.log(2 * jnp.pi) + log_det_Sig)  # (K,)

    def maha_sq_of_comp(L_k, mu_k):
        z_mu_k = z - mu_k
        y = solve_triangular(L_k, z_mu_k.T, lower=True)
        return jnp.sum(y**2, axis=0)

    maha_sq = jax.vmap(maha_sq_of_comp, in_axes=(0, 0))(L, mu).T  # (N, K)

    log_phi = jnp.log(phi + eps)                     # avoid log(0)
    logp    = log_phi[None, :] + const[None, :] - 0.5 * maha_sq  # (N, K)

    log_prob  = logsumexp(logp, axis=1)  # (N,)
    energies  = -log_prob

    return energies


def calc_prf(energy, labels):
    threshold = jnp.percentile(energy, 100 - 20)
    print(f'Threshold: {threshold}')
    y_hat = (energy > threshold).astype(jnp.int32)
    y_true = labels.astype(jnp.int32)
    precision, recall, f1, _ = prf(y_true, y_hat, average="binary")
    return precision, recall, f1
