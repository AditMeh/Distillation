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
        return softmax_student


def distillation_loss(student_softmax, teacher_softmax, ground_truth):
    cross_entropy_loss = nn.NLLLoss()
    return loss
