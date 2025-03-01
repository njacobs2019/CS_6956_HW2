"""
This script trains several different ensemble members on poly dataset with FGE
"""

import os

import comet_ml
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split

from .datasets import Poly
from .models import ModelSmall
from .train import train, train_fge

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")


### Params
epochs = 400
device = torch.device("cuda:1")


# Dataset
ds = Poly(10000)
train_ds, test_ds = random_split(ds, [0.8, 0.2])
train_loader = DataLoader(train_ds, batch_size=2000, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=2000, shuffle=False)

model = ModelSmall(input_dim=1, hidden_dim=64)
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
        name="POLY_FGE_TEST",
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
    DataLoader(train_ds, batch_size=256, shuffle=True, drop_last=True),
    test_loader,
    num_members=5,
    comet_experiment=experiment,
    alpha_1=8e-3,
    alpha_2=4e-5,
)
