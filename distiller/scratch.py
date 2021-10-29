
import torch

a = torch.ones(size=(32, 10))
b = torch.ones(size=(32, 1))

print(torch.nn.CrossEntropyLoss(a, b).shape)