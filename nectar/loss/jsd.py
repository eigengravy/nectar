"""
Jensen-Shannon Divergence Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JSDLoss(nn.Module):
    def __init__(self, gamma=0.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.kld = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, student_logits, teacher_logits):
        """Forward pass."""
        # p = F.log_softmax(student_logits, dim=-1)
        # q = F.log_softmax(teacher_logits, dim=-1)
        # m = 0.5 * (p + q)
        # jsd_loss = 0.5 * (self.kld(m, p) + self.kld(m, q))
        # return self.gamma * jsd_loss
        p = student_logits.view(-1, student_logits.size(-1)).log_softmax(-1)
        with torch.no_grad():
            q = teacher_logits.view(-1, teacher_logits.size(-1)).log_softmax(-1)
        m = 0.5 * (p + q)
        jsd_loss = 0.5 * (self.kld(m, p) + self.kld(m, q))
        return jsd_loss


# class JSDLoss(nn.Module):
#     def __init__(self, num_classes=200, tau=3, beta=1):
#         super(JSDLoss, self).__init__()
#         self.num_classes = num_classes
#         self.tau = tau
#         self.beta = beta
#         self.ce = nn.CrossEntropyLoss()
#         self.kld = nn.KLDivLoss(reduction="batchmean", log_target=True)

#     def forward(self, local_logits, global_logits):
#         p = local_logits.view(-1, local_logits.size(-1)).log_softmax(-1)
#         with torch.no_grad():
#             q = global_logits.view(-1, global_logits.size(-1)).log_softmax(-1)
#         m = 0.5 * (p + q)
#         jsd_loss = 0.5 * (self.kld(m, p) + self.kl(m, q))
#         return jsd_loss
