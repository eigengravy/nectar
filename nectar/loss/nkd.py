import torch
import torch.nn as nn
import torch.nn.functional as F


class NKDLoss(nn.Module):
    def __init__(
        self,
        temp=1.0,
        gamma=1.5,
    ):
        super(NKDLoss, self).__init__()

        self.temp = temp
        self.gamma = gamma
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, student_logits, teacher_logits, labels):

        if len(labels.size()) > 1:
            label = torch.max(labels, dim=1, keepdim=True)[1]
        else:
            label = labels.view(len(labels), 1)

        # N*class
        N, c = student_logits.shape
        s_i = self.log_softmax(student_logits)
        t_i = F.softmax(teacher_logits, dim=1)
        # N*1
        s_t = torch.gather(s_i, 1, label)
        t_t = torch.gather(t_i, 1, label).detach()

        loss_t = -(t_t * s_t).mean()

        mask = torch.ones_like(student_logits).scatter_(1, label, 0).bool()
        student_logits = student_logits[mask].reshape(N, -1)
        teacher_logits = teacher_logits[mask].reshape(N, -1)

        # N*class
        S_i = self.log_softmax(student_logits / self.temp)
        T_i = F.softmax(teacher_logits / self.temp, dim=1)

        loss_non = (T_i * S_i).sum(dim=1).mean()
        loss_non = -self.gamma * (self.temp**2) * loss_non

        return loss_t + loss_non
