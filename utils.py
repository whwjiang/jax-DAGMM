import jax.numpy as jnp


def cosine_similarity(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    norm_x = jnp.linalg.norm(x, axis=1, keepdims=True)
    norm_y = jnp.linalg.norm(y, axis=1, keepdims=True)
    dot_product = jnp.sum(norm_x * norm_y, axis=1)
    return dot_product


def euclidean_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(x - y, ord=2, axis=1) / jnp.linalg.norm(x, ord=2, axis=1)
