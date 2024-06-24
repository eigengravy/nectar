import argparse
from collections import OrderedDict
import copy
import csv
from datetime import datetime
import json
import os
import pickle
import time
from typing import Dict, Tuple, List
import uuid

import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import Metrics
from flwr.common.typing import Scalar

from datasets import Dataset
from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from tqdm import tqdm

# from nectar.models.tiny_imagenet import apply_transforms, get_dataset
# from nectar.models.tiny_imagenet.vgg16 import VGG16 as Net, train
# from nectar.strategy.mifl import MIFL
# from nectar.utils.model import test
from nectar.loss.kl import DistillLoss
from nectar.loss.ntd import NTDLoss
from nectar.strategy.mifl import MIFL
from nectar.utils.mi import mutual_information, normalized_mutual_information
from nectar.utils.params import get_params, set_params

from transformers import AutoModelForSequenceClassification

from flwr_datasets.partitioner import DirichletPartitioner

from evaluate import load as load_metric
from transformers import AdamW


import random
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "distilbert-base-uncased"


def load_centralized_testset():
    # fds = FederatedDataset(dataset="imdb", partitioners={"train": 10})
    # partition = fds.load_split("test")
    # tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, model_max_length=512)

    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], truncation=True)

    # partition_train_test = partition.map(tokenize_function, batched=True)
    # partition_train_test = partition_train_test.remove_columns("text")
    # partition_train_test = partition_train_test.rename_column("label", "labels")

    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # testloader = DataLoader(partition, batch_size=32, collate_fn=data_collator)

    # return testloader

    raw_datasets = load_dataset("dair-ai/emotion", "split")
    # raw_datasets = raw_datasets.shuffle(seed=42)
    # remove unnecessary data split
    # del raw_datasets["unsupervised"]
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    # We will take a small sample in order to reduce the compute time, this is optional
    # train_population = random.sample(range(len(raw_datasets["train"])), 100)
    # test_population = random.sample(range(len(raw_datasets["test"])), 100)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    # tokenized_datasets["train"] = tokenized_datasets["train"].select(train_population)
    # tokenized_datasets["test"] = tokenized_datasets["test"].select(test_population)
    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # trainloader = DataLoader(
    #     tokenized_datasets["train"],
    #     shuffle=True,
    #     batch_size=32,
    #     collate_fn=data_collator,
    # )
    testloader = DataLoader(
        tokenized_datasets["test"], batch_size=32, collate_fn=data_collator
    )
    return testloader


def load_data(partition_id):
    """Load IMDB data (training and eval)"""
    fds = FederatedDataset(
        dataset="dair-ai/emotion",
        subset="split",
        partitioners={
            "train": DirichletPartitioner(
                num_partitions=10, alpha=0.5, partition_by="label"
            ),
        },
    )
    partition = fds.load_partition(partition_id)

    # Divide data: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, model_max_length=512)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    partition_train_test = partition_train_test.map(tokenize_function, batched=True)
    partition_train_test = partition_train_test.remove_columns("text")
    partition_train_test = partition_train_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    trainloader = DataLoader(
        partition_train_test["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator,
    )

    testloader = DataLoader(
        partition_train_test["test"], batch_size=32, collate_fn=data_collator
    )

    return trainloader, testloader


def train(student, trainloader, optim, epochs, device):
    device = DEVICE
    optim = AdamW(student.parameters(), lr=5e-5)
    start = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    teacher = copy.deepcopy(student)
    teacher.to(device)
    teacher.eval()
    student.to(device)
    student.train()
    # distiller = NTDLoss(temp=5.0, gamma=0.5)
    distiller = DistillLoss(temp=3.0, gamma=0.5)
    # distiller = CosineLoss(gamma=0.5)
    # distiller = JSDLoss(gamma=0.5)
    # distiller = NKDLoss()
    # distiller = RKDLoss()
    mi_gauss, mi_cat = 0, 0
    for _ in range(epochs):
        for batch in tqdm(trainloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            student_outputs = student(**batch)
            student_logits = student_outputs.logits
            with torch.no_grad():
                teacher_outputs = teacher(**batch)
                teacher_logits = teacher_outputs.logits
            # print("DEBUG", student_logits, teacher_logits)
            # print("DEBUG", teacher_logits.shape, type(teacher_logits))
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

            ce_loss = student_outputs.loss
            # dist_loss = distiller(student_logits, teacher_logits, batch["labels"])
            dist_loss = distiller(student_logits, teacher_logits)
            print(f"CE Loss: {ce_loss.item()}, Distill Loss: {dist_loss.item()}")
            # print("DEBUG", 1000 * dist_loss.item())
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
            optim.zero_grad()

    train_loss, train_acc = test(student, trainloader)
    results = {
        "loss": train_loss,
        "accuracy": train_acc,
        "mi_gauss": mi_gauss / len(trainloader),
        "mi_cat": mi_cat / len(trainloader),
        "t_diff": time.time() - start,
    }
    return results


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in tqdm(testloader):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, cid):
        self.trainset = trainset
        self.cid = cid
        self.valset = valset
        self.model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT, num_labels=2
        ).to(DEVICE)

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]
        # trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)
        trainloader = self.trainset
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        results = train(
            self.model, self.trainset, optimizer, epochs=epochs, device=self.device
        )
        return get_params(self.model), len(trainloader), results

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        # valloader = DataLoader(self.valset, batch_size=64)
        valloader = self.valset
        loss, accuracy = test(self.model, valloader)
        return float(loss), len(valloader), {"accuracy": float(accuracy)}


def get_client_fn():
    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        # client_dataset = dataset.load_partition(int(cid), "train")
        trainloader, testloader = load_data(int(cid))
        # Now let's split it into train (90%) and validation (10%)
        # client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

        # trainset = client_dataset_splits["train"]
        # valset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        # trainset = trainset.with_transform(apply_transforms)
        # valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(trainloader, testloader, cid).to_client()

    return client_fn


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 2,  # Number of local epochs done by clients
        "batch_size": 64,  # Batch size to use by clients during fit()
    }
    return config


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "accuracy": sum(accuracies) / sum(examples),
    }


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [m["accuracy"] for num_examples, m in metrics]
    loss = [m["loss"] / num_examples for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    mi_gauss = [m["mi_gauss"] / num_examples for num_examples, m in metrics]
    mi_cat = [m["mi_cat"] / num_examples for num_examples, m in metrics]

    metrics = {
        "accuracy": accuracies,
        "loss": loss,
        "mi_gauss": mi_gauss,
        "mi_cat": mi_cat,
    }

    print(metrics)
    return metrics


def get_evaluate_fn():
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

        model = AutoModelForSequenceClassification.from_pretrained(
            CHECKPOINT, num_labels=2
        ).to(DEVICE)

        set_params(model, parameters)
        model.to(device)

        # testset = centralized_testset.with_transform(apply_transforms)
        disable_progress_bar()

        # testloader = DataLoader(testset, batch_size=50)
        testloader = load_centralized_testset()
        loss, accuracy = test(model, testloader)

        return loss, {"accuracy": accuracy}

    return evaluate


def save_history(history, run_id):
    if history.losses_distributed:
        with open(f"runs/{run_id}/losses_distributed.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "loss"])
            writer.writerows(history.losses_distributed)

    if history.losses_centralized:
        with open(f"runs/{run_id}/losses_centralized.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["round", "loss"])
            writer.writerows(history.losses_centralized)

    if history.metrics_distributed_fit:
        for metric in history.metrics_distributed_fit:
            with open(f"runs/{run_id}/metrics_distributed_fit_{metric}.csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow(["round", metric])
                writer.writerows(history.metrics_distributed_fit[metric])

    if history.metrics_distributed:
        for metric in history.metrics_distributed:
            with open(f"runs/{run_id}/metrics_distributed_{metric}.csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow(["round", metric])
                writer.writerows(history.metrics_distributed[metric])

    if history.metrics_centralized:
        for metric in history.metrics_centralized:
            with open(f"runs/{run_id}/metrics_centralized_{metric}.csv", "a+") as f:
                writer = csv.writer(f)
                writer.writerow(["round", metric])
                writer.writerows(history.metrics_centralized[metric])

    with open(f"runs/{run_id}/history.txt", "w") as f:
        f.write(repr(history))

    pickle.dump(
        history,
        open(f"runs/{run_id}/history.pkl", "wb"),
    )


def main():
    parser = argparse.ArgumentParser(description="Nectar")
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of CPUs to assign to a virtual client",
    )
    parser.add_argument(
        "--num_gpus",
        type=float,
        default=0.0,
        help="Ratio of GPU memory to assign to a virtual client",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="",
        help="Description of the simulation run",
    )
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset name")
    parser.add_argument(
        "--partition_by", type=str, default="label", help="Partition the dataset by"
    )
    parser.add_argument(
        "--split_train",
        type=str,
        choices=["niid", "iid"],
        default="niid",
        help="Data distribution",
    )
    parser.add_argument(
        "--split_test",
        type=str,
        choices=["niid", "iid"],
        default="iid",
        help="Data distribution",
    )
    parser.add_argument("--train", type=str, default="train", help="Train split name")
    parser.add_argument("--test", type=str, default="test", help="Test split name")
    # Simulation arguments
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, default=2, help="Number of rounds")

    # MIFL arguments
    parser.add_argument(
        "--critical_value", type=float, default=0.25, help="Critical value"
    )
    parser.add_argument("--mi_type", type=str, default="mi_cat", help="MI type")

    run_id = datetime.now().strftime("%d-%b-%H%M") + "-" + str(uuid.uuid4())[:8]

    args = parser.parse_args()

    # mnist_fds, centralized_testset = get_dataset(args.num_clients)

    # Configure the strategy
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1,  # Sample 10% of available clients for training
    #     fraction_evaluate=1,  # Sample 5% of available clients for evaluation
    #     min_available_clients=2,
    #     on_fit_config_fn=fit_config,
    #     evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Aggregate federated metrics
    #     evaluate_fn=get_evaluate_fn(),  # Global evaluation function
    #     fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    # )

    strategy = MIFL(
        fraction_fit=1,  # Sample 10% of available clients for training
        fraction_evaluate=1,  # Sample 5% of available clients for evaluation
        min_available_clients=args.num_clients,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(),  # Global evaluation function
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        critical_value=args.critical_value,
        mi_type=args.mi_type,
    )

    # client = fl.client.ClientApp(
    #     client_fn=get_client_fn(mnist_fds),
    # )

    # # ServerApp for Flower-Next
    # server = fl.server.ServerApp(
    #     config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    #     strategy=strategy,
    # )

    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    config = {
        **args.__dict__,
        "run_id": run_id,
        **fit_config(0),
        "start_time": datetime.now().strftime("%H:%M:%S"),
    }

    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(),
        num_clients=args.num_clients,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
    )

    config["end_time"] = datetime.now().strftime("%H:%M:%S")

    os.makedirs(f"runs/{run_id}")

    with open(f"runs/{run_id}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    save_history(history, run_id)


if __name__ == "__main__":
    main()
