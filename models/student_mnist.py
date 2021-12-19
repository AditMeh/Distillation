"""
Taken from https://github.com/pytorch/examples/blob/master/mnist/main.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentNetMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28, 14*14)
        self.fc2 = nn.Linear(14*14, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
