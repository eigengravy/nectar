import torch
from nectar.utils.mi import mutual_information
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from flwr_datasets import FederatedDataset


def train_fn(
    student,
    teacher,
    optimizer,
    criterion,
    distiller,
    trainloader,
    epochs,
    device,
    distill=True,
):
    student.train()
    student.to(device)
    teacher.eval()
    teacher.to(device)
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            optimizer.zero_grad()
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)
            loss = criterion(student_logits, labels)
            if distill:
                loss += distiller(student_logits, teacher_logits, labels)
            loss.backward()
            optimizer.step()
    train_loss, train_acc = test_fn(student, trainloader, device)
    return {
        "loss": train_loss,
        "accuracy": train_acc,
    }


def mi_fn(student, teacher, testloader, mi_type, device):
    student.eval()
    student.to(device)
    teacher.eval()
    teacher.to(device)
    mi = 0
    with torch.no_grad():
        for batch in testloader:
            images, _ = batch["img"].to(device), batch["fine_label"].to(device)
            student_logits = student(images)
            teacher_logits = teacher(images)
            mi += mutual_information(student_logits, teacher_logits, mi_type=mi_type)
    return mi.item()


def test_fn(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["fine_label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_dataset(partitioners, batch_size=64, test_size=0.1):
    fds = FederatedDataset(
        dataset="cifar100",
        partitioners=partitioners,
    )

    def apply_transforms(batch):

        batch["img"] = [
            Compose(
                [
                    ToTensor(),
                    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                    Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0) == 1 else x),
                ]
            )(img)
            for img in batch["img"]
        ]
        return batch

    testloader = DataLoader(
        fds.load_split("test").with_transform(apply_transforms), batch_size=batch_size
    )

    def get_client_loader(cid: str):
        client_dataset = fds.load_partition(int(cid), "train")
        client_dataset_splits = client_dataset.train_test_split(
            test_size=test_size, seed=42
        )
        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]
        trainloader = DataLoader(
            trainset.with_transform(apply_transforms), batch_size=batch_size
        )
        valloader = DataLoader(
            valset.with_transform(apply_transforms), batch_size=batch_size
        )
        return trainloader, valloader

    return testloader, get_client_loader
