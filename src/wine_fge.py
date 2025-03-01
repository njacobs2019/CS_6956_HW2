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
from .train import train, train_fge

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")


### Params
batch_size = 128
epochs = 200
h_dim = 32
device = torch.device("cuda:1")


# Dataset
train_ds, test_ds = get_wine_data()
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Model
model = ModelSmall(input_dim=11, hidden_dim=h_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)

# Log and train
experiment = comet_ml.start(
    api_key=COMET_API_KEY,
    project_name="pdl-hw2",
    mode="create",
    online=True,
    experiment_config=comet_ml.ExperimentConfig(
        auto_metric_logging=False,
        disabled=False,  # Set True for debugging runs
        name="WINE_FGE_TEST",
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

# RUN FGE
train_fge(
    model,
    device,
    DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
    test_loader,
    num_members=5,
    comet_experiment=experiment,
    # alpha_1=8e-3,
    alpha_1=0.05,
    # alpha_2=4e-5,
    alpha_2=4e-4,
)
