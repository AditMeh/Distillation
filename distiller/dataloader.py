from torchvision.datasets import MNIST, mnist
import torch
from torchvision import transforms
from utils import oneHotEncoder


def create_dataloaders_mnist():
    training_dataset = torch.utils.data.DataLoader(MNIST(root='./data', train=True, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]), target_transform=transforms.Compose([oneHotEncoder(10)])), batch_size=32, shuffle=True)

    test_set = torch.utils.data.DataLoader(MNIST(root='./data', train=False, download=True, transform=transforms.Compose(
        [transforms.ToTensor()]), target_transform=transforms.Compose([transforms.ToTensor(), oneHotEncoder(10)])), batch_size=32, shuffle=True)

    return training_dataset, test_set


if __name__ == "__main__":

    t, _ = create_dataloaders_mnist()

    for x, y in t:
        print(y.shape)
        break
