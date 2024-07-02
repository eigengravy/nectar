"""
Similarity-Preserving Knowledge Distillation
https://arxiv.org/pdf/1907.09682.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, student_logits, teacher_logits, labels=None):
        student_logits = student_logits.view(student_logits.size(0), -1)
        G_s = torch.mm(student_logits, student_logits.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        teacher_logits = teacher_logits.view(teacher_logits.size(0), -1)
        G_t = torch.mm(teacher_logits, teacher_logits.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss
