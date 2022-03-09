# engine file where the optimizer, 
# train and test functions are defined

import torch
from torch import nn as nn

import model
import config
import dataset

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.model.parameters(), 
    lr=config.Learning_rate
    )

# Define training function

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, corect = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(config.DEVICE), y.to(config.DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            corect += (pred.argmax(1) ==y).type(torch.float).sum().item()

    test_loss /= num_batches
    corect /= size

    print(
        f"Test Error: \n Accuracy: {(100*corect): >0.1f}%, Avg loss: {test_loss:>8f} \n")
    