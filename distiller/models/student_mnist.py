"""
Taken from https://github.com/pytorch/examples/blob/master/mnist/main.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StudentNetMnist(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1*28*28, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

if __name__ == "__main__":
    net = StudentNetMnist()
    x = torch.ones(size=(1, 1, 28, 28))

    print(net(x).shape)


