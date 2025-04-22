import jax.numpy as jnp
from flax import nnx
from jax.scipy.linalg import inv, cholesky

import utils


class DAGMM(nnx.Module):
    def __init__(
        self,
        n_features=122,
        n_components=4,
        lambda_1=0.1,
        lambda_2=0.005,
        *,
        rngs: nnx.Rngs
    ):
        self.N = n_features
        self.k = n_components
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        # encoder network
        self.cn1 = nnx.Linear(self.N, 60, rngs=rngs)
        self.cn2 = nnx.Linear(60, 30, rngs=rngs)
        self.cn3 = nnx.Linear(30, 10, rngs=rngs)
        self.cn4 = nnx.Linear(10, 1, rngs=rngs)
        # decoder network
        self.cn5 = nnx.Linear(1, 10, rngs=rngs)
        self.cn6 = nnx.Linear(10, 30, rngs=rngs)
        self.cn7 = nnx.Linear(30, 60, rngs=rngs)
        self.cn8 = nnx.Linear(60, self.N, rngs=rngs)
        # estimation network
        self.en1 = nnx.Linear(3, 10, rngs=rngs)
        self.en2 = nnx.Dropout(0.5, rngs=rngs)
        self.en3 = nnx.Linear(10, self.k, rngs=rngs)

    def __encode(self, x):
        x = nnx.tanh(self.cn1(x))
        x = nnx.tanh(self.cn2(x))
        x = nnx.tanh(self.cn3(x))
        x = self.cn4(x)
        return x

    def __decode(self, x):
        x = nnx.tanh(self.cn5(x))
        x = nnx.tanh(self.cn6(x))
        x = nnx.tanh(self.cn7(x))
        x = self.cn8(x)
        return x

    def __estimate(self, x):
        x = nnx.tanh(self.en1(x))
        x = self.en2(x)
        x = self.en3(x)
        x = nnx.softmax(x)
        return x

    def __call__(self, x):
        z_c = self.__encode(x)
        x_hat = self.__decode(z_c)
        z_r_1 = utils.euclidean_distance(x, x_hat).reshape(-1, 1)
        # print(f'z_r_1.shape: {z_r_1.shape}')
        # print(f'z_r_1: {z_r_1}')
        z_r_2 = utils.cosine_similarity(x, x_hat).reshape(-1, 1)
        # print(f'z_r_2.shape: {z_r_2.shape}')
        # print(f'z_r_2: {z_r_2}')
        z = jnp.concat((z_r_1, z_r_2, z_c), axis=1)
        # print(f'z.shape: {z.shape}')
        # print(f'z: {z}')
        gamma = self.__estimate(z)
        return gamma, x_hat, z

    @staticmethod
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

    def calc_sample_energy(self, z, phi, mu, covs):
        # broadcast differences
        z_mu = jnp.expand_dims(z, 1) - jnp.expand_dims(mu, 0)  # Shape: (N, K, D)

        eps = 1e-12
        cov_invs = []
        cov_dets = []

        # Loop over components and calculate the covariance inverse and determinant
        for i in range(self.k):
            cov_i = covs[i] + (jnp.eye(covs[i].shape[-1]) * eps)  # numerical stability
            cov_inv_i = inv(cov_i)
            cov_invs.append(jnp.expand_dims(cov_inv_i, 0))  # Shape: (1, D, D)

            L = cholesky(cov_i * (2 * jnp.pi), lower=True)
            cov_det_i = jnp.prod(jnp.diagonal(L)) ** 2
            cov_dets.append(jnp.expand_dims(cov_det_i, 0))  # Shape: (1,)

        cov_invs = jnp.concatenate(cov_invs, axis=0)  # Shape: (K, D, D)
        cov_dets = jnp.expand_dims(
            jnp.sqrt(jnp.concatenate(cov_dets)), 0
        )  # Shape: (K, 1)

        E_z = jnp.sum(jnp.expand_dims(z_mu, -1) * jnp.expand_dims(cov_invs, 0), axis=-2)
        E_z = jnp.sum(E_z * z_mu, axis=-1)
        E_z = jnp.exp(-0.5 * E_z)
        E_z = -jnp.log(jnp.sum(jnp.expand_dims(phi, 0) * E_z / cov_dets, axis=1) + eps)
        return jnp.mean(E_z)
