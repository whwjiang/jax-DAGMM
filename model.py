import jax.numpy as jnp
from flax import nnx

import utils

class DAGMM(nnx.Module):
    # this is the exact model architecture the paper authors 
    # used for evaluation on the KDDCUP99 dataset
    def __init__(self, *, rngs: nnx.Rngs):
        self.k = 4 
        # encoder network
        self.cn1 = nnx.Linear(120, 60, rngs=rngs)
        self.cn2 = nnx.Linear(60, 30, rngs=rngs)
        self.cn3 = nnx.Linear(30, 10, rngs=rngs)
        self.cn4 = nnx.Linear(10, 1, rngs=rngs)
        # decoder network
        self.cn5 = nnx.Linear(1, 10, rngs=rngs)
        self.cn6 = nnx.Linear(10, 30, rngs=rngs)
        self.cn7 = nnx.Linear(30, 60, rngs=rngs)
        self.cn8 = nnx.Linear(60, 120, rngs=rngs)
        # estimation network
        self.en1 = nnx.Linear(3, 10, rngs=rngs)
        self.en2 = nnx.Dropout(0.5, rngs=rngs)
        self.en3 = nnx.Linear(10, 4, rngs=rngs)
    
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
        x = nnx.softmax(x)
        return x
    
    def __call__(self, x):
        z_c = self.__encode(x)
        x_hat = self.__decode(z_c)
        z_r_1 = utils.euclidean_distance(x, x_hat)
        z_r_2 = utils.cosine_similarity(x, x_hat)
        z = jnp.concat(jnp.atleast_1d(z_r_1), jnp.atleast_1d(z_r_2), z_c)
        gamma = self.__estimate(z)
        return gamma, x_hat, z_c
