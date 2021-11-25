import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.modules import loss
from torch.nn import CrossEntropyLoss


def distillation_loss(student_logits, T, teacher_logits, ground_truth_probs, weight):

    return ((weight * T**2 * CrossEntropyLoss()(student_logits/T, F.softmax(teacher_logits/T, dim=1)))
            + (1-weight) * (CrossEntropyLoss()(student_logits, ground_truth_probs)))
