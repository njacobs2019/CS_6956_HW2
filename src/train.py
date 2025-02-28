"""
Implements training routine
"""

import comet_ml
import comet_ml.integration
import comet_ml.integration.pytorch
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(  # pylint: disable=R0913,R0917,R0914
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    comet_experiment: comet_ml.Experiment | None,
    optuna_trial: optuna.trial.Trial | None = None,
    log_batches: bool = True,
) -> float:
    """
    Trains a single model with MSE loss.
    Trains and logs the model to Comet. Then evaluates the model on the validation set.
    Allows for hyperparameter tuning with Optuna.

    Args:
        model (nn.Module): Torch model
        optimizer (torch.optim.Optimizer): Torch optimizer
        device (torch.device): Computing device
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Number of epochs.
        comet_experiment (comet_ml.Experiment): Comet experiment for logging
        trial (optuna.trial.Trial | None, optional): Optuna trial for tuning.
                                                     Defaults to None.
        log_batches (bool, optional): Log stats after every mini-batch. Default, True.

    Raises:
        optuna.TrialPruned: If using Optuna with pruning

    Returns:
        float: final_val_loss
    """

    # Train the model
    model.to(device)
    loss_fn = nn.MSELoss()

    step = 0  # current training batch_num
    for epoch in tqdm(range(epochs), unit="Epoch"):
        # Training loop
        model.train()
        train_loss = 0.0
        total_samples = 0

        # for x, y in tqdm(train_loader, desc="Training", leave=False, unit="Batch"):
        for x, y in train_loader:
            batch_size = x.size(0)
            total_samples += batch_size

            optimizer.zero_grad()

            # Forward pass
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            loss = loss_fn(y, prediction)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update loss metric
            train_loss += loss.item() * batch_size
            if comet_experiment is not None and log_batches:
                comet_experiment.log_metric("train_loss", loss.item(), step=step)
            step += 1

        train_loss = train_loss / total_samples

        # Validation loop
        model.eval()
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            # for x, y in tqdm(val_loader, desc="Validation", leave=False, unit="Batch"):
            for x, y in val_loader:
                batch_size = x.size(0)
                total_samples += batch_size

                x = x.to(device)
                y = y.to(device)

                # Calculate loss
                prediction = model(x)
                loss = loss_fn(y, prediction)

                val_loss += loss.item() * batch_size

        val_loss = val_loss / total_samples

        # Log the loss
        if comet_experiment is not None:
            comet_experiment.log_metrics(
                {"train_loss": train_loss, "val_loss": val_loss}, epoch=epoch
            )

        # Log results to optuna for hyperparameter tuning and pruning
        if optuna_trial:
            optuna_trial.report(val_loss, epoch)
            if optuna_trial.should_prune():
                raise optuna.TrialPruned()

    # Log the trained model
    if comet_experiment is not None:
        print("Saving weights to disk")
        torch.save(
            model.to("cpu").state_dict(),
            f"./checkpoints/{comet_experiment.get_key()}.pt",
        )

    return val_loss
