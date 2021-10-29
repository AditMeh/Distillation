import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules import loss

def cross_entropy(input, target):
    return (-(target * torch.log(input))).sum() / input.shape[0]


def distillation_loss(student_softmax, teacher_softmax, ground_truth, weight):
    ce_loss = weight * cross_entropy(student_softmax, teacher_softmax) + (
        1-weight) * cross_entropy(student_softmax, teacher_softmax)
    return ce_loss


