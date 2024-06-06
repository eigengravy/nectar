import argparse
from collections import OrderedDict
import csv
from datetime import datetime
import json
import os
import pickle
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

from nectar.models.tiny_imagenet import apply_transforms, get_dataset
from nectar.models.tiny_imagenet.vgg16 import VGG16 as Net, train
from nectar.strategy.mifl import MIFL
from nectar.utils.model import test
from nectar.utils.params import get_params, set_params


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainset, valset, cid):
        self.trainset = trainset
        self.cid = cid
        self.valset = valset
        self.model = Net()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device

    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)

        # Read from config
        batch, epochs = config["batch_size"], config["epochs"]
        trainloader = DataLoader(self.trainset, batch_size=batch, shuffle=True)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        results = train(
            self.model, trainloader, optimizer, epochs=epochs, device=self.device
        )
        return get_params(self.model), len(trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_params(self.model, parameters)
        valloader = DataLoader(self.valset, batch_size=64)
        loss, accuracy = test(self.model, valloader, device=self.device)
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def get_client_fn(dataset: FederatedDataset):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(int(cid), "train")

        # Now let's split it into train (90%) and validation (10%)
        client_dataset_splits = client_dataset.train_test_split(test_size=0.1, seed=42)

        trainset = client_dataset_splits["train"]
        valset = client_dataset_splits["test"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)
        valset = valset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(trainset, valset, cid).to_client()

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


def get_evaluate_fn(
    centralized_testset: Dataset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_params(model, parameters)
        model.to(device)

        testset = centralized_testset.with_transform(apply_transforms)
        disable_progress_bar()

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

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

    run_id = datetime.now().strftime("%d-%b-%H%M") + "-" + str(uuid.uuid4())[:8]

    args = parser.parse_args()

    mnist_fds, centralized_testset = get_dataset(args.num_clients)

    # Configure the strategy
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=1,  # Sample 10% of available clients for training
    #     fraction_evaluate=1,  # Sample 5% of available clients for evaluation
    #     min_available_clients=2,
    #     on_fit_config_fn=fit_config,
    #     evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Aggregate federated metrics
    #     evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
    #     fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    # )

    strategy = MIFL(
        fraction_fit=1,  # Sample 10% of available clients for training
        fraction_evaluate=1,  # Sample 5% of available clients for evaluation
        min_available_clients=2,
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,  # Aggregate federated metrics
        evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        critical_value=0.25,
        mi_type="mi_gauss",
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
        client_fn=get_client_fn(mnist_fds),
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
