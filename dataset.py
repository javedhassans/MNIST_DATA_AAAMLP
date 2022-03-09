import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import config


# downloading dataset from open datasets
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform = ToTensor(),
)

# download test data from open dataser
test_data = datasets.FashionMNIST(
    root= "data",
    train=False,
    download=True,
    transform=ToTensor(),
)



# create dataloaders
train_dataloader = DataLoader(training_data, batch_size=config.batch_size)
test_dataloader = DataLoader(training_data, batch_size=config.batch_size)


# for X, y in test_dataloader:
#     print(f"shape of X [N, C, H, W] : {X.shape}")
#     print(f"shape of y: {y.shape} {y.dtype}")