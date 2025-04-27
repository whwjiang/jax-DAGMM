import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader
import torch
import os


class KDDCupData:
    def __init__(self, key, data_dir, mode, indices_file, overwrite=False):
        """Loading the data for train and test."""
        data = jnp.load(data_dir, allow_pickle=True)

        labels = data["kdd"][:, -1]
        features = data["kdd"][:, :-1]
        # non-anomalous data is the data that is not labeled "normal."
        normal_data = features[labels == 0]
        normal_labels = labels[labels == 0]

        if overwrite or not os.path.exists(indices_file):
            # Generate and save indices if they don't exist
            n_train = int(normal_data.shape[0] * 0.5)
            ixs = jax.random.permutation(key, normal_data.shape[0])
            np.savez(indices_file, train_ixs=ixs[:n_train], test_ixs=ixs[n_train:])

        indices = np.load(indices_file)
        train_ixs = indices["train_ixs"]
        test_ixs = indices["test_ixs"]

        if mode == "train":
            self.x = normal_data[train_ixs]
            self.y = normal_labels[train_ixs]
        elif mode == "test":
            anomalous_data = features[labels == 1]
            anomalous_labels = labels[labels == 1]
            self.x = jnp.concatenate((anomalous_data, normal_data[test_ixs]), axis=0)
            self.y = jnp.concatenate((anomalous_labels, normal_labels[test_ixs]), axis=0)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return np.float32(self.x[index]), np.float32(self.y[index])


def get_dataloader(key, batch_size, mode, 
                   data_dir="../data/kddcup.npz", 
                   indices_file="../data/indices.npz",
                   overwrite=False):
    """Returning dataloader."""
    data = KDDCupData(key, data_dir, mode, indices_file=indices_file, overwrite=overwrite)
    dataloader = DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=0
    )
    return dataloader

def cleanup_dataloader(dataloader):
    if hasattr(dataloader, "_iterator"):
        try:
            dataloader._iterator._shutdown_workers()
        except Exception:
            pass  # if it’s already shut down or not present, ignore

    dataloader = None
    torch.cuda.empty_cache()
