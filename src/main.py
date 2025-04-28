import jax
import jax.numpy as jnp
import optax
from flax import nnx
from model import DAGMM
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors
from numpy import random

from utils import calc_prf
from dataloader import get_dataloader
from train import train
from eval import eval


# Does training and evaluation of the DAGMM model on the KDDCup dataset.
# Saves the evaluation results to a file.
def main():
    key = jax.random.PRNGKey(random.randint(0, 2**32))
    batch_size = 1024
    key, dataloader_key = jax.random.split(key, 2)

    model = DAGMM(n_features=118, rngs=nnx.Rngs(key))
    dataloader_train = get_dataloader(
        dataloader_key, batch_size=batch_size, mode="train", overwrite=True
    )
    dataloader_test = get_dataloader(None, batch_size=batch_size, mode="test")

    learning_rate = 0.0001
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    train(model, optimizer, metrics, dataloader_train, epochs=200, save_model=True)
    z, energy, labels = eval(model, dataloader_train, dataloader_test)

    jnp.savez("../graphs/eval.npz", z=z, energy=energy, labels=labels)
    print(f'Processing done. Use process.ipynb to visualize the results.')
    precision, recall, f1 = calc_prf(energy, labels)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Save the plot instead of showing it interactively
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(z[:, 2], z[:, 0], z[:, 1], c=labels, cmap="cool", s=1)
    ax.set_xlabel("Encoded")
    ax.set_ylabel("Euclidean")
    ax.set_zlabel("Cosine")

    mplcursors.cursor(sc, hover=True)
    plt.savefig("../graphs/eval.png")
    print("Plot saved to ../graphs/eval.png")


if __name__ == "__main__":
    main()
