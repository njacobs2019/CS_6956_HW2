"""
Utilities for working with ensembles
"""

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn


class Result(NamedTuple):
    """
    Holds ensemble prediction statistics
    """

    mean: Tensor
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
        min=torch.min(predictions, dim=0)[0],
        max=torch.max(predictions, dim=0)[0],
        sd=torch.std(predictions, dim=0),
    )
