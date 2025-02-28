"""
Utilities for working with ensembles
"""

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


class Result(NamedTuple):
    """
    Holds ensemble prediction statistics
    """

    mean: Tensor
    median: Tensor
    min: Tensor
    max: Tensor
    sd: Tensor


def predict(x: Tensor, models: Sequence[nn.Module]) -> Result:
    """
    Returns inference statistics
    Assumes x and models are on the same device

    Args:
        x (Tensor): Model input
        device (torch.device): Device to run the model on
        models (Sequence[nn.Module]): List of models

    Results:
        result (Result): Tensors on CPU
    """

    predictions_list: list[Tensor] = []

    # Inference
    with torch.no_grad():
        for model in models:
            y = model(x)
            predictions_list.append(y)

    predictions = torch.stack(predictions_list, dim=0)

    return Result(
        mean=torch.mean(predictions, dim=0),
        median=torch.median(predictions, dim=0).values,
        min=torch.min(predictions, dim=0)[0],
        max=torch.max(predictions, dim=0)[0],
        sd=torch.std(predictions, dim=0),
    )


def evaluate(
    loader: DataLoader, device: torch.device, ensemble: Sequence[nn.Module]
) -> tuple[float, float, float]:
    """
    Evaluates performance and uncertainty of an ensemble
    MSE_mean - MSE when ensemble prediction is mean of ensemble
    MSE_median - MSE when ensemble prediction is median of ensemble

    Args:
        loader (DataLoader): Data loader
        device (torch.device): Compute device
        ensemble (Sequence[nn.Module]): Ensemble

    Returns:
        tuple[float, float]: MSE_mean, MSE_median, average SD
    """

    mse_mean = 0.0
    mse_median = 0.0
    sd = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            batch_size = x.size(0)
            total_samples += batch_size

            x = x.to(device)
            y = y.to(device)
            pred = predict(x, ensemble)

            mse_update = torch.mean(torch.square(pred.mean - y))
            mse_median_update = torch.mean(torch.square(pred.median - y))
            sd_update = torch.mean(pred.sd)

            mse_mean += mse_update.item() * batch_size
            mse_median += mse_median_update.item() * batch_size
            sd += sd_update.item() * batch_size

    mse_mean = mse_mean / total_samples
    mse_median = mse_median / total_samples
    sd = sd / total_samples

    return mse_mean, mse_median, sd
