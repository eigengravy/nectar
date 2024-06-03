"""
Jensen-Shannon Divergence Loss.
"""

import torch.nn as nn
import torch.nn.functional as F


class JSDLoss(nn.Module):
    def __init__(self, gamma=0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.kld = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits):
        """Forward pass."""
        p = F.log_softmax(student_logits, dim=-1)
        q = F.log_softmax(teacher_logits, dim=-1)
        m = 0.5 * (p + q)
        jsd_loss = 0.5 * (self.kld(m, p) + self.kld(m, q))
        return self.gamma * jsd_loss
