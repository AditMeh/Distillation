import torch
from torch._C import dtype


class oneHotEncoder(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, batch_labels):
        target_onehot = torch.zeros(10)
        target_onehot.scatter(0, torch.LongTensor([batch_labels]), 1.0)
        return target_onehot
