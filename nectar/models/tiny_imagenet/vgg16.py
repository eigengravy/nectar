import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import copy

from nectar.loss.cosine import CosineLoss
from nectar.loss.dkd import DKDLoss
from nectar.loss.jsd import JSDLoss
from nectar.loss.kl import DistillLoss
from nectar.loss.ntd import NTDLoss
from nectar.loss.nkd import NKDLoss
from nectar.loss.rkd import RKDLoss
from nectar.utils.mi import normalized_mutual_information, mutual_information
from nectar.utils.model import test


class VGG16(nn.Module):
    def __init__(self, num_classes=200) -> None:
        super(VGG16, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(2048, 2048)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(
            F.batch_norm(
                self.conv1(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv2(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv3(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv4(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv5(x), running_mean=None, running_var=None, training=True
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv6(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv7(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv8(x), running_mean=None, running_var=None, training=True
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv9(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv10(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv11(x), running_mean=None, running_var=None, training=True
            )
        )
        x = F.relu(
            F.batch_norm(
                self.conv12(x), running_mean=None, running_var=None, training=True
            )
        )
        x = self.pool(
            F.relu(
                F.batch_norm(
                    self.conv13(x), running_mean=None, running_var=None, training=True
                )
            )
        )
        x = x.view(-1, 2048)
        x = F.relu(self.fc(F.dropout(x, 0.5)))
        x = F.relu(self.fc1(F.dropout(x, 0.5)))
        x = self.fc2(x)
        return x


def train(student, trainloader, optim, epochs, device: str):
    start = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    teacher = copy.deepcopy(student)
    teacher.to(device)
    teacher.eval()
    student.to(device)
    student.train()
    distiller = NTDLoss(temp=5.0, gamma=0.5)
    # distiller = DistillLoss(temp=3.0, gamma=0.5)
    # distiller = CosineLoss(gamma=0.5)
    # distiller = JSDLoss(gamma=0.5)
    # distiller = NKDLoss()
    # distiller = RKDLoss()
    mi_gauss, mi_cat = 0, 0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optim.zero_grad()
            student_logits = student(images)
            teacher_logits = teacher(images)

            with torch.no_grad():
                mi_gauss += (
                    mutual_information(
                        student_logits, teacher_logits, dist_type="gaussian"
                    )
                    .sum()
                    .item()
                )
                mi_cat += (
                    normalized_mutual_information(
                        student_logits, teacher_logits, dist_type="categorical"
                    )
                    .sum()
                    .item()
                )

            ce_loss = criterion(student_logits, labels)
            dist_loss = distiller(student_logits, teacher_logits, labels)
            # dist_loss = distiller(student_logits, teacher_logits)
            print(f"CE Loss: {ce_loss.item()}, Distill Loss: {dist_loss.item()}")
            loss = ce_loss + dist_loss
            # loss = criterion(student_logits, labels) + distiller(
            #     student_logits, teacher_logits, labels
            # )
            # loss = dist_loss

            # one_hot_labels = F.one_hot(labels, num_classes=200).float()
            # gamma = 0.5
            # loss = (1 - gamma) * mutual_information(
            #     teacher_logits, one_hot_labels, dist_type="gaussian"
            # ).sum() + gamma * mutual_information(
            #     student_logits, one_hot_labels, dist_type="gaussian"
            # ).sum()

            loss.backward()
            optim.step()

    train_loss, train_acc = test(student, trainloader, device)
    results = {
        "loss": train_loss,
        "accuracy": train_acc,
        "mi_gauss": mi_gauss / len(trainloader.dataset),
        "mi_cat": mi_cat / len(trainloader.dataset),
        "t_diff": time.time() - start,
    }
    return results
