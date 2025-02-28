"""
This script trains several different traditional ensemble members on the poly dataset
"""

import os

import comet_ml
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split

from .datasets import Poly
from .models import PolyModelMicro
from .train import train

load_dotenv()
COMET_API_KEY = os.getenv("COMET_API_KEY")


### Params
batch_size = 2000
epochs = 1000
device = torch.device("cuda:1")


# Dataset
ds = Poly(10000)
train_ds, test_ds = random_split(ds, [0.8, 0.2])
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

for i in range(10):
    # Model
    model = PolyModelMicro(input_dim=1, hidden_dim=16)
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
            # name=f"Poly_Trad_Ensemble_Micro_Member_{epochs}_epochs",
            name=f"Poly_Trad_Ensemble_Micro_Member_{epochs}_16_dim",
        ),
    )

    train(model, optimizer, device, train_loader, test_loader, epochs, experiment)
