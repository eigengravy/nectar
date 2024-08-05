from typing import List, Tuple
from flwr.common import Metrics
from logging import INFO, WARNING
from flwr.common.logger import log
import numpy as np


def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    client_accuracy = [m["accuracy"] for _, m in metrics]
    client_loss = [m["loss"] for _, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples),
        "client_accuracy": ",".join(map(str, client_accuracy)),
        "client_loss": ",".join(map(str, client_loss)),
    }


def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    mi = [num_examples * m["mi"] for num_examples, m in metrics]

    client_accuracies = [m["accuracy"] for _, m in metrics]
    client_losses = [m["loss"] for _, m in metrics]
    client_mi = [m["mi"] for _, m in metrics]
    client_cid = ",".join([str(m["cid"]) for _, m in metrics])

    log(INFO, f"Clients {client_cid}")
    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples),
        "mi": np.nansum(mi) / sum(examples),
        "client_accuracy": ",".join(map(str, client_accuracies)),
        "client_loss": ",".join(map(str, client_losses)),
        "client_mi": ",".join(map(str, client_mi)),
        "client_cid": client_cid,
    }
