"""
Preservation of the Global Knowledge by Not-True Distillation in Federated Learning
Gihun Lee, Minchan Jeong, Yongjin Shin, Sangmin Bae, Se-Young Yun
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NTDLoss(nn.Module):
    def __init__(self, temp, gamma) -> None:
        super().__init__()
        self.temp = temp
        self.gamma = gamma
        self.kld = nn.KLDivLoss(reduction="batchmean")

    def _refine_as_not_true(self, logits, targets):
        nt = torch.arange(0, self.num_classes).to(logits.device)
        nt = nt.repeat(logits.size(0), 1)
        nt = nt[nt[:, :] != targets.view(-1, 1)]
        nt = nt.view(-1, self.num_classes - 1)
        logits = torch.gather(logits, 1, nt)
        return logits

    def forward(self, student_logits, teacher_logits, labels):
        """Forward pass."""
        soft_targets = F.softmax(
            self._refine_as_not_true(teacher_logits, labels) / self.temp, dim=-1
        )
        soft_prob = F.log_softmax(
            self._refine_as_not_true(student_logits, labels) / self.temp, dim=-1
        )
        distill_loss = (self.temp**2) * self.kld(soft_prob, soft_targets)
        return self.gamma * distill_loss
