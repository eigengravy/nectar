"""
"""

import torch.nn as nn
import torch.nn.functional as F


class CosineLoss(nn.Module):
    def __init__(self, gamma) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CosineEmbeddingLoss()

    def forward(self, student_rep, teacher_rep):
        """Forward pass."""
        ce_loss = self.ce(student_rep, teacher_rep)
        return self.gamma * ce_loss
