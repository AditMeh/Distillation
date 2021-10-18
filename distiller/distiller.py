import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules import loss


class Distiller(nn.Module):
    def __init__(self, student_network: nn.Module, teacher_network: nn.Module, temperature: int):
        super(Distiller, self).__init__()

        self.student = student_network
        self.teacher = teacher_network
        self.temperature = temperature

    def forward(self, x):
        # Both the student and teacher outputs should be logits
        student_logits = self.student(x)

        with torch.no_grad():
            teacher_logits = self.teacher(x)
            softmax_teacher = nn.LogSoftmax(teacher_logits/self.temperature)

        softmax_student = nn.LogSoftmax(student_logits/self.temperature)
        return softmax_student, softmax_teacher


def cross_entropy(input, target):
    # Input is a log-probability
    return -(target * input).sum() / input.shape[0]


def distillation_loss_with_hard_target(student_logsoftmax, teacher_logsoftmax, ground_truth, weight):
    ce_loss = weight * cross_entropy(student_logsoftmax, teacher_logsoftmax) + (
        1-weight) * cross_entropy(student_logsoftmax, ground_truth)
    return ce_loss


def distillation_loss(student_logsoftmax, teacher_logsoftmax):
    ce_loss = cross_entropy(student_logsoftmax, teacher_logsoftmax)
    return ce_loss


if __name__ == "__main__":
    x = np.ones((1, 3))
    y = np.array([[0.1, 0.1, 0.1]])
    loss = distillation_loss(x, y)
    print(loss)
