import jax.numpy as jnp
from flax import nnx

from utils import cosine_similarity, euclidean_distance


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

        z_r_1 = euclidean_distance(x, x_hat).reshape(-1, 1)
        z_r_2 = cosine_similarity(x, x_hat).reshape(-1, 1)
        z = jnp.concat((z_r_1, z_r_2, z_c), axis=1)

        gamma = self.__estimate(z)
        return gamma, x_hat, z
