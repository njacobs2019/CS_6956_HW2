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


def train_fge(  # pylint: disable=R0913,R0917,R0914,R0915
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_members: int = 10,
    alpha_1: float = 0.01,
    alpha_2: float = 0.0005,
    comet_experiment: comet_ml.Experiment | None = None,
):
    """
    Takes in an initialized/partially trained model and runs FGE to calcualte multiple
    checkpoints.  It uses a cyclic learning rate.

    Args:
        model (nn.Module): The model
        device (torch.device): Compute device
        train_loader (DataLoader): Training dataloader
        val_loader (DataLoader): Validation dataloader
        num_members (int, optional): Number of ensemble members. Defaults to 10.
        alpha_1 (float, optional): High learning rate. Defaults to 0.01.
        alpha_2 (float, optional): Low learning rate. Defaults to 0.0005.
        comet_experiment (comet_ml.Experiment | None, optional): Comet experiment for
            logging. Defaults to None.
    """

    epoch_per_cycle = 4

    iters_per_half_cycle = int(len(train_loader) * epoch_per_cycle / 2)

    print(f"Num training batches: {len(train_loader)}")
    print(f"Iters per half cycle {iters_per_half_cycle}")

    optimizer = torch.optim.Adam(model.parameters(), alpha_1, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=alpha_1,
        max_lr=alpha_2,
        step_size_up=iters_per_half_cycle,
        step_size_down=iters_per_half_cycle,
    )

    # Train the model
    model.to(device)
    loss_fn = nn.MSELoss()

    end_condition = False
    model_num = 0
    step = 0  # current batchnum in training
    # for epoch in tqdm(range(num_members * epoch_per_cycle)):
    for epoch in range(num_members * epoch_per_cycle):
        model.train()
        train_loss = 0.0
        total_samples = 0

        # Train loop
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
            scheduler.step()

            # Update loss metric
            train_loss += loss.item() * batch_size
            if comet_experiment is not None:
                comet_experiment.log_metrics(
                    {
                        "fge_train_loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=step,
                )

            train_loss = train_loss / total_samples

            # if proper step, save the model
            if abs(optimizer.param_groups[0]["lr"] - alpha_2) < 0.000001:
                print(f"Saving model {model_num} to disk, step {step}")
                if comet_experiment is not None:
                    torch.save(
                        model.state_dict(),
                        f"./checkpoints/{comet_experiment.get_key()}_fge_{model_num}.pt",
                    )
                model_num += 1
                if model_num == num_members:
                    end_condition = True
                    break
            step += 1

        # Validation loop
        model.eval()
        val_loss = 0.0
        total_samples = 0

        with torch.no_grad():
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
                {"fge_train_loss": train_loss, "fge_val_loss": val_loss}, epoch=epoch
            )

        if end_condition:
            break
