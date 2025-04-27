import jax
import jax.numpy as jnp
import optax
from flax import nnx
from model import DAGMM

from sklearn.metrics import precision_recall_fscore_support as prf, accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mplcursors

from dataloader import get_dataloader
from train import train
from eval import eval

# Does training, eval, and plotting
def main():
    key = jax.random.PRNGKey(42)
    batch_size = 1024
    key, dataloader_key = jax.random.split(key, 2)

    model = DAGMM(n_features=118, rngs=nnx.Rngs(key))
    dataloader_train = get_dataloader(dataloader_key, batch_size=batch_size, mode='train')
    dataloader_test = get_dataloader(None, batch_size=batch_size, mode='test')
    
    learning_rate = 0.0001
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate))
    metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average('loss')
    )
    
    train(model, optimizer, metrics, dataloader_train, epochs=200, save_model=True)
    z, energy, labels = eval(model, dataloader_train, dataloader_test)
    
    # Do statistics 
    threshold = jnp.percentile(energy, 100 - 20)
    print(f'Threshold: {threshold}')
    y_hat = (energy > threshold).astype(jnp.int32)
    y_true = labels.astype(jnp.int32)
    print(f'Accuracy: {accuracy_score(y_true, y_hat)}')
    precision, recall, f1, _ = prf(y_true, y_hat, average='binary')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1: {f1}')
    
    # plot it
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(z[:, 1], z[:, 0], z[:, 2], c=labels, cmap='plasma', s=1)
    ax.set_xlabel('Encoded')
    ax.set_ylabel('Euclidean')
    ax.set_zlabel('Cosine')

    # Add interactivity with mplcursors
    mplcursors.cursor(sc, hover=True)

    # Show plot
    plt.show()

if __name__ == "__main__":
    main()