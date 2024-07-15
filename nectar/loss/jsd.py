"""
Jensen-Shannon Divergence Loss.
"""

import torch
import torch.nn as nn


class JSDLoss(nn.Module):
    def __init__(self, gamma=0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.kld = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, student_logits, teacher_logits, labels=None):
        p = student_logits.view(-1, student_logits.size(-1)).log_softmax(-1)
        with torch.no_grad():
            q = teacher_logits.view(-1, teacher_logits.size(-1)).log_softmax(-1)
        m = 0.5 * (p + q)
        jsd_loss = 0.5 * (self.kld(m, p) + self.kld(m, q))
        return jsd_loss
