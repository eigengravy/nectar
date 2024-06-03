import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from nectar.loss.kl import DistillLoss
from nectar.utils.mi import mutual_information
from nectar.utils.model import test


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(student, trainloader, optim, epochs, device: str):
    criterion = torch.nn.CrossEntropyLoss()
    teacher = copy.deepcopy(student)
    teacher.to(device)
    teacher.eval()
    student.to(device)
    student.train()
    distiller = DistillLoss(temp=3.0, gamma=0.5)
    mi_gauss, mi_cat = 0, 0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optim.zero_grad()
            student_logits = student(images)
            teacher_logits = teacher(images)
            loss = criterion(student_logits, labels) + distiller(
                student_logits, teacher_logits
            )
            loss.backward()
            optim.step()

            with torch.no_grad():
                mi_gauss += (
                    mutual_information(
                        student_logits, teacher_logits, dist_type="gaussian"
                    )
                    .sum()
                    .item()
                )
                mi_cat += (
                    mutual_information(
                        student_logits, teacher_logits, dist_type="categorical"
                    )
                    .sum()
                    .item()
                )

    train_loss, train_acc = test(student, trainloader, device)
    results = {
        "loss": train_loss,
        "accuracy": train_acc,
        "mi_gauss": mi_gauss,
        "mi_cat": mi_cat,
    }
    return results
