import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    model.train()
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' %
              (num_correct, num_samples, 100 * acc))


class ResBlock(nn.Module):
    '''
    Re-implement residual block in [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

    '''

    def __init__(self, in_channel, bottleneck_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, bottleneck_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(bottleneck_channel, in_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        a1 = self.relu1(self.bn1(self.conv1(x)))
        a2 = self.bn2(self.conv2(a1))
        out = self.relu2(a2 + x)

        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


if __name__ == "__main__":
    device = torch.device('cuda') \
        if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32

    model = nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=1, padding=3),
        ResBlock(64, 32),
        ResBlock(64, 32),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # NC H/2 W/2
        ResBlock(128, 64),
        ResBlock(128, 64),
        nn.Conv2d(128, 10, kernel_size=3, stride=2, padding=1),  # NC H/4 W/4
        nn.AvgPool2d(8),
        nn.Flatten()
    )

    model = model.to(device=device)

    # x = torch.zeros((5, 3, 32, 32), dtype=dtype, device=device)
    # out = model(x)
    # print(out.shape)
    # print(list(model.modules()))

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module)
    learning_rate, wd = 1e-3, 1e-7
    epochs = 3

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=wd)
    criteron = F.cross_entropy

    # Training loop...
