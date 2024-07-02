"""
A wrapper around the flower History class that offers centralized and distributed metrics per timestamp instead of per round.
It also groups distributed_fit metrics per client instead of per client instead of per round.

metrics_centralized: {
    "accuracy": [ (timestamp1, value1) , .... ]
}
metrics_distributed: {
    "client_ids": [ (timestamp1, [cid1, cid2, cid3]) ... ]
    "accuracy": [ (timestamp1, [value1, value2, value3]) , .... ]
} 
metrics_distributed_fit: {
    "accuracy": { 
        cid1: [
            (timestamp1, value1), 
            (timestamp2, value2), 
            (timestamp3, value3)
            ...
            ],
        ...
    }
    ...
}
# Metrics collected after each merge into the global model. (Global model evaluated centrally after merge.)
metrics_centralized_async: {
    "accuracy": [ (timestamp1, value1) , .... ]
}
Note: value1 is collected at timestamp1 in metrics_distributed_fit.
"""

from flwr.server.history import History
from threading import Lock
from functools import reduce
from typing import Dict, List, Tuple

from flwr.common.typing import Scalar


class AsyncHistory(History):

    def __init__(self) -> None:
        self.metrics_distributed_fit_async = {}
        self.metrics_centralized_async = {}
        # ^ metrics aggregated after each merge into the global model.
        self.losses_centralized_async = []
        super().__init__()

    def add_metrics_distributed_fit_async(
        self, client_id: str, metrics: Dict[str, Scalar], timestamp: float
    ) -> None:
        """Add metrics entries (from distributed fit)."""
        lock = Lock()
        with lock:
            for key in metrics:
                if key not in self.metrics_distributed_fit_async:
                    self.metrics_distributed_fit_async[key] = {}
                if client_id not in self.metrics_distributed_fit_async[key]:
                    self.metrics_distributed_fit_async[key][client_id] = []
                self.metrics_distributed_fit_async[key][client_id].append(
                    (timestamp, metrics[key])
                )

    def add_metrics_centralized_async(
        self, metrics: Dict[str, Scalar], timestamp: float
    ) -> None:
        """Add metrics entries (from centralized evaluation)."""
        lock = Lock()
        with lock:
            for metric in metrics:
                if metric not in self.metrics_centralized_async:
                    self.metrics_centralized_async[metric] = []
                self.metrics_centralized_async[metric].append(
                    (timestamp, metrics[metric])
                )

    def add_loss_centralized_async(self, timestamp: float, loss: float) -> None:
        """Add loss entries (from centralized evaluation)."""
        lock = Lock()
        with lock:
            self.losses_centralized_async.append((timestamp, loss))

    def add_loss_centralized(self, timestamp: float, loss: float) -> None:
        return super().add_loss_centralized(timestamp, loss)

    def add_loss_distributed(self, timestamp: float, loss: float) -> None:
        return super().add_loss_distributed(timestamp, loss)

    def add_metrics_centralized(
        self, timestamp: float, metrics: Dict[str, bool | bytes | float | int | str]
    ) -> None:
        return super().add_metrics_centralized(timestamp, metrics)

    def add_metrics_distributed(
        self, timestamp: float, metrics: Dict[str, bool | bytes | float | int | str]
    ) -> None:
        return super().add_metrics_distributed(timestamp, metrics)

    def __repr__(self) -> str:
        """Create a representation of History.

        The representation consists of the following data (for each round) if present:

        * distributed loss.
        * centralized loss.
        * distributed training metrics.
        * distributed evaluation metrics.
        * centralized metrics.

        Returns
        -------
        representation : str
            The string representation of the history object.
        """
        rep = ""
        lock = Lock()
        with lock:
            if self.losses_distributed:
                rep += "History (loss, distributed):\n" + reduce(
                    lambda a, b: a + b,
                    [
                        f"\tround {server_round}: {loss}\n"
                        for server_round, loss in self.losses_distributed
                    ],
                )
            if self.losses_centralized:
                rep += "\nHistory (loss, centralized):\n" + reduce(
                    lambda a, b: a + b,
                    [
                        f"\tround {server_round}: {loss}\n"
                        for server_round, loss in self.losses_centralized
                    ],
                )
            if self.metrics_distributed_fit:
                rep += "\nHistory (metrics, distributed, fit):\n" + str(
                    self.metrics_distributed_fit
                )
            if self.metrics_distributed:
                rep += "\nHistory (metrics, distributed, evaluate):\n" + str(
                    self.metrics_distributed
                )
            if self.metrics_centralized:
                rep += "\nHistory (metrics, centralized):\n" + str(
                    self.metrics_centralized
                )
            if self.metrics_distributed_fit_async:
                rep += "History (metrics, distributed, fit, async):\n" + str(
                    self.metrics_distributed_fit_async
                )
            if self.metrics_centralized_async:
                rep += "\nHistory (metrics, distributed, evaluate, async):\n" + str(
                    self.metrics_centralized_async
                )
            if self.losses_centralized_async:
                rep += "\nHistory (loss, centralized, async):\n" + str(
                    self.losses_centralized_async
                )
            return rep
