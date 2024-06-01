"""
Distilling the Knowledge in a Neural Network
Geoffrey Hinton, Oriol Vinyals, Jeff Dean
"""

import torch.nn as nn
import torch.nn.functional as F


class DistillLoss(nn.Module):
    def __init__(self, temp, gamma) -> None:
        super().__init__()
        self.temp = temp
        self.gamma = gamma
        self.kld = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        """Forward pass."""
        soft_targets = F.softmax(teacher_logits / self.temp, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temp, dim=-1)
        distill_loss = (self.temp**2) * self.kld(soft_prob, soft_targets)
        return self.gamma * distill_loss
