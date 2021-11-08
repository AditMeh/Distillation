import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules import loss
from torch.nn import CrossEntropyLoss


def distillation_loss(student_logits, teacher_softmax, ground_truth_probs, weight):
    return ((weight * CrossEntropyLoss()(student_logits, teacher_softmax))
            + (1-weight) * (CrossEntropyLoss()(student_logits, ground_truth_probs)))


if __name__ == "__main__":
    teacher = torch.Tensor(
        np.array([0.5, 0.01, 0.01, 0.23, 0.25])).unsqueeze(0)
    student = torch.Tensor(
        np.array([0.5, 0.01, 0.01, 0.23, 0.25])).unsqueeze(0)

    gt = torch.Tensor([1.0, 0.0, 0.0, 0.0, 0.0])

    print(distillation_loss()(student, teacher))
