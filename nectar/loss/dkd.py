import torch
import torch.nn as nn
import torch.nn.functional as F


class DKDLoss(nn.Module):

    def __init__(
        self,
        temp=4.0,
        alpha=1.0,
        beta=6.0,
    ):
        super(DKDLoss, self).__init__()
        self.temp = temp
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def _get_mask(logit, gt_label, value):
        gt_label = gt_label.reshape(-1)
        mask = torch.zeros_like(logit).scatter_(1, gt_label.unsqueeze(1), value).bool()
        return mask

    @staticmethod
    def _cat_mask(t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def forward(self, logit_s, logit_t, gt_label):
        gt_mask = self._get_mask(logit_s, gt_label, 1)
        other_mask = self._get_mask(logit_s, gt_label, 0)
        pred_student = F.softmax(logit_s / self.temp, dim=1)
        pred_teacher = F.softmax(logit_t / self.temp, dim=1)
        pred_student = self._cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = self._cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = torch.log(pred_student)
        tckd_loss = (
            F.kl_div(log_pred_student, pred_teacher, size_average=False)
            * (self.temp**2)
            / gt_label.shape[0]
        )
        pred_teacher_part2 = F.softmax(logit_t / self.temp - 1000.0 * gt_mask, dim=1)
        log_pred_student_part2 = F.log_softmax(
            logit_s / self.temp - 1000.0 * gt_mask, dim=1
        )
        nckd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
            * (self.temp**2)
            / gt_label.shape[0]
        )

        return self.alpha * tckd_loss + self.beta * nckd_loss
