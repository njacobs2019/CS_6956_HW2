"""
Datasets
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

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
