from torchvision import datasets
from torchvision.datasets import MNIST
import torch
from torch._C import dtype
from torchvision import transforms
from torchvision.transforms.transforms import ToTensor
from utils import OneHotEncoder

from PIL import Image

from visualization.plot_train_graph import display_image_grid


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, classes, classwise_dict, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.classes = classes

        self.classwise_dict = classwise_dict

        self.image_tensor, self.labels_tensor = self.generate_dataset(classes)

    def __len__(self):
        assert len(self.image_tensor) == len(self.labels_tensor)
        return len(self.image_tensor)

    def __getitem__(self, idx):
        image = self.image_tensor[idx]

        image = Image.fromarray(image.detach().numpy(), mode='L')
        label = self.labels_tensor[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(image)

        return image, label

    def generate_dataset(self, classes):
        subsampled_dataset_imgs = []
        subsampled_dataset_labels = []

        for c in classes:
            subsampled_dataset_imgs.extend(self.classwise_dict[c])
            subsampled_dataset_labels.extend(torch.ones(
                size=(len(self.classwise_dict[c]), ), dtype=torch.int64)*c)

        return torch.stack(subsampled_dataset_imgs), torch.hstack(subsampled_dataset_labels)


def generate_mnist_classwise_dict(processed_mnist):
    images, labels = processed_mnist

    classwise_dict = {i: [] for i in range(10)}

    for i, label in enumerate(labels):
        classwise_dict[label.item()].append(images[i])

    return classwise_dict


def create_onehot_dataloaders_mnist(classes_train=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                    classes_val=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]):

    training_dataset = torch.utils.data.DataLoader(
        MNISTDataset(data_directory='data/MNIST/processed/training.pt',
                     classes=classes_train,
                     target_transform=transforms.Compose([OneHotEncoder(10)]),
                     classwise_dict=classwise_dict),
        batch_size=32, shuffle=True)

    test_set = torch.utils.data.DataLoader(
        MNISTDataset(data_directory='data/MNIST/processed/test.pt',
                     classes=classes_val,
                     transform=ToTensor(),
                     target_transform=transforms.Compose([OneHotEncoder(10)]),
                     classwise_dict=classwise_dict),
        batch_size=32, shuffle=True)

    return training_dataset, test_set


def create_dataloaders_mnist(classwise_dict_train, classwise_dict_val, classes_train):

    training_dataset = torch.utils.data.DataLoader(
        MNISTDataset(classes=classes_train,
                     transform=ToTensor(),
                     classwise_dict=classwise_dict_train),
        batch_size=32, shuffle=True)

    test_set = torch.utils.data.DataLoader(
        MNISTDataset(classes=[i for i in range(10)],
                     transform=ToTensor(),
                     classwise_dict=classwise_dict_val),
        batch_size=32, shuffle=True)

    return training_dataset, test_set
