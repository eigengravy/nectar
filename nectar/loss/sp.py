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

    def forward(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)

        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)

        loss = F.mse_loss(norm_G_s, norm_G_t)

        return loss
