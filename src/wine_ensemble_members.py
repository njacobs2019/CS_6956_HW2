"""
This script trains several different traditional ensemble members on the wine dataset
"""

import os

import comet_ml
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader

from .datasets import get_wine_data
from .models import ModelSmall
from .train import train

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")


### Params
batch_size = 256
epochs = 600
h_dim = 32
device = torch.device("cuda:1")


# Dataset
train_ds, test_ds = get_wine_data()
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

for i in range(10):
    # Model
    model = ModelSmall(input_dim=11, hidden_dim=h_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0)

    # Log and train
    experiment = comet_ml.start(
        api_key=COMET_API_KEY,
        project_name="pdl-hw2",
        mode="create",
        online=True,
        experiment_config=comet_ml.ExperimentConfig(
            auto_metric_logging=False,
            disabled=False,  # Set True for debugging runs
            name=f"Wine_Trad_Ensemble_Member_{epochs}_{h_dim}",
        ),
    )

    train(
        model,
        optimizer,
        device,
        train_loader,
        test_loader,
        epochs,
        experiment,
        log_batches=False,
    )
