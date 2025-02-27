"""
Datasets
"""

import numpy as np
import sklearn
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset
from ucimlrepo import fetch_ucirepo

np.random.seed(0)


class Poly(Dataset):
    """
    2D polynomial regression dataset
    """

    def __init__(self, n: int, sd: float = 0.2):
        """
        Args:
            n (int): Size of dataset
            sd (float, optional): Standard deviation of noise. Defaults to 0.2.
        """
        x = np.random.triangular(-2, 0, 2, size=n)
        y = self.func(x) + np.random.normal(0, sd, x.shape)
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y).to(torch.float32)

    def func(self, x: NDArray):
        """
        Calculates the polynomial
        """
        return (x - 1) * (x + 1) * x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[i].unsqueeze(0), self.y[i].unsqueeze(0)


def get_wine_data() -> tuple[Dataset, Dataset]:
    """
    Returns training and testing datasets for the wine data

    Returns:
        tuple[Dataset, Dataset]: train, test
    """

    # Fetch dataset
    wine_quality = fetch_ucirepo(id=186)
    X: NDArray = wine_quality.data.features.values.astype(
        np.float32
    )  # samples x features
    y: NDArray = wine_quality.data.targets.values.astype(np.float32)  # samples x 1

    # Split the dataset
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Scale features
    scaler = sklearn.preprocessing.StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train_scaled), torch.from_numpy(y_train)
    )
    test_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test_scaled), torch.from_numpy(y_test)
    )

    return train_ds, test_ds


if __name__ == "__main__":
    train, test = get_wine_data()
