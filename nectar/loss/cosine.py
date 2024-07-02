"""
Cosine Loss
"""

import torch.nn as nn


class CosineLoss(nn.Module):
    def __init__(self, gamma) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CosineEmbeddingLoss()

    def forward(self, student_logits, teacher_logits, labels):
        ce_loss = self.ce(student_logits, teacher_logits, labels)
        return self.gamma * ce_loss
