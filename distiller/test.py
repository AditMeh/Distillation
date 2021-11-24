from torchvision.datasets import MNIST
import torch
from torchvision import transforms
from utils import OneHotEncoder



def create_dataloaders_mnist():
    training_dataset = torch.utils.data.DataLoader(MNIST(root='./data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor()])), batch_size=32, shuffle=True)

    test_set = torch.utils.data.DataLoader(MNIST(root='./data', train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor()])), batch_size=32, shuffle=True)

    return training_dataset, test_set


tt, ts = create_dataloaders_mnist()

torch.unique(next(iter(tt))[0])