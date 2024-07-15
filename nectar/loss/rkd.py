import torch
import torch.nn as nn
import torch.nn.functional as F


"""
From https://github.com/lenscloth/RKD/blob/master/metric/loss.py
"""


class RKDLoss(nn.Module):
    """
    Relational Knowledge Distillation
    https://arxiv.org/pdf/1904.05068.pdf
    """

    def __init__(self, w_dist=3.0, w_angle=3.0):
        super().__init__()

        self.w_dist = w_dist
        self.w_angle = w_angle

    def forward(self, student_logits, teacher_logits):
        loss = self.w_dist * self.rkd_dist(
            student_logits, teacher_logits
        ) + self.w_angle * self.rkd_angle(student_logits, teacher_logits)

        return loss

    def rkd_dist(self, student_logits, teacher_logits, labels=None):
        teacher_logits_dist = self.pdist(teacher_logits, squared=False)
        mean_teacher_logits_dist = teacher_logits_dist[teacher_logits_dist > 0].mean()
        teacher_logits_dist = teacher_logits_dist / mean_teacher_logits_dist

        student_logits_dist = self.pdist(student_logits, squared=False)
        mean_student_logits_dist = student_logits_dist[student_logits_dist > 0].mean()
        student_logits_dist = student_logits_dist / mean_student_logits_dist

        loss = F.smooth_l1_loss(student_logits_dist, teacher_logits_dist)

        return loss

    def rkd_angle(self, student_logits, teacher_logits):
        # N x C --> N x N x C
        teacher_logits_vd = teacher_logits.unsqueeze(0) - teacher_logits.unsqueeze(1)
        norm_teacher_logits_vd = F.normalize(teacher_logits_vd, p=2, dim=2)
        teacher_logits_angle = torch.bmm(
            norm_teacher_logits_vd, norm_teacher_logits_vd.transpose(1, 2)
        ).view(-1)

        student_logits_vd = student_logits.unsqueeze(0) - student_logits.unsqueeze(1)
        norm_student_logits_vd = F.normalize(student_logits_vd, p=2, dim=2)
        student_logits_angle = torch.bmm(
            norm_student_logits_vd, norm_student_logits_vd.transpose(1, 2)
        ).view(-1)

        loss = F.smooth_l1_loss(student_logits_angle, teacher_logits_angle)

        return loss

    def pdist(self, feat, squared=False, eps=1e-12):
        student_logitsquare = feat.pow(2).sum(dim=1)
        feat_prod = torch.mm(feat, feat.t())
        feat_dist = (
            student_logitsquare.unsqueeze(0)
            + student_logitsquare.unsqueeze(1)
            - 2 * feat_prod
        ).clamp(min=eps)

        if not squared:
            feat_dist = feat_dist.sqrt()

        feat_dist = feat_dist.clone()
        feat_dist[range(len(feat)), range(len(feat))] = 0

        return feat_dist
