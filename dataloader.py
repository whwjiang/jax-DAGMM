import jax
import jax.numpy as jnp
import numpy as np
from torch.utils.data import DataLoader


class KDDCupData:
    def __init__(self, key, data_dir, mode):
        """Loading the data for train and test."""
        data = jnp.load(data_dir, allow_pickle=True)

        labels = data["kdd"][:, -1]
        features = data["kdd"][:, :-1]
        # non-anomalous data is the data that is not labeled "normal."
        normal_data = features[labels == 0]
        normal_labels = labels[labels == 0]

        n_train = int(normal_data.shape[0] * 0.5)
        ixs = jax.random.permutation(key, normal_data.shape[0])
        normal_data_test = normal_data[ixs[n_train:]]
        normal_labels_test = normal_labels[ixs[n_train:]]

        if mode == "train":
            self.x = normal_data[ixs[:n_train]]
            self.y = normal_labels[ixs[:n_train]]
        elif mode == "test":
            anomalous_data = features[labels == 1]
            anomalous_labels = labels[labels == 1]
            self.x = jnp.concatenate((anomalous_data, normal_data_test), axis=0)
            self.y = jnp.concatenate((anomalous_labels, normal_labels_test), axis=0)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return np.float32(self.x[index]), np.float32(self.y[index])


def load_kddcup99(key, batch_size, data_dir="./data/kddcup.npz"):
    """Returning train and test dataloaders."""
    train = KDDCupData(key, data_dir, "train")
    dataloader_train = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=0
    )

    test = KDDCupData(key, data_dir, "test")
    dataloader_test = DataLoader(
        test, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return dataloader_train, dataloader_test
