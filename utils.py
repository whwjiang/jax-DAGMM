import jax.numpy as jnp

def cosine_similarity(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the cosine similarity between two vectors.
    
    Args:
        x (jnp.ndarray): First vector.
        y (jnp.ndarray): Second vector.
    
    Returns:
        jnp.ndarray: Cosine similarity (scalar) between the two input vectors.
    """
    dot_product = jnp.dot(x, y)
    norm_x = jnp.linalg.norm(x)
    norm_y = jnp.linalg.norm(y)
    epsilon = 1e-10  # small constant to avoid division-by-zero
    return dot_product / (norm_x * norm_y + epsilon)

def euclidean_distance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Euclidean distance between two vectors normalized by the first.
    
    Args:
        x (jnp.ndarray): First vector.
        y (jnp.ndarray): Second vector.
    
    Returns:
        jnp.ndarray: Normalized Euclidean distance between the two input vectors.
    """
    return jnp.linalg.norm(x - y) / jnp.linalg.norm(x)