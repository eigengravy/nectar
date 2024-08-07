import math
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from flwr.server.client_proxy import ClientProxy
from logging import INFO, WARNING
from hydra.utils import instantiate
from flwr.common.logger import log
from flwr.common import (
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
import numpy as np
from scipy.signal import savgol_filter
from flwr.server.client_manager import ClientManager


class DynaMIFL(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        mi_type: str,
        trigger_round: int = None,
        low_critical_value: float = 0.05,
        high_critical_value: float = 0.20,
        window_length: int = 10,
        optimize: bool = True,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            inplace=True,
        )
        self.mi_type = mi_type
        self.trigger_round = trigger_round
        self.low_critical_value = low_critical_value
        self.high_critical_value = high_critical_value
        self.has_triggered = False
        self.mi_history = []
        self.window_length = window_length
        self.client_score = [0] * min_available_clients
        self.optimize = optimize

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"DynMIFL(mi_type={self.mi_type})"
        return rep

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        if not self.accept_failures and failures:
            return None, {}

        results = [result for result in results if result[1].metrics["cid"] != -1]

        # Collect mutual information of clients from fit results
        mi = [fit_res.metrics["mi"] for _, fit_res in results]

        # # If not enough clients have mutual information, fill with average
        # if len(mi) < self.min_fit_clients:
        #     mi.extend([sum(mi) / len(mi)] * (self.min_fit_clients - len(mi)))

        # Save weighted average of mutual information
        self.mi_history.append(
            sum(
                [
                    res.num_examples * res.metrics["mi"]
                    for _, res in results
                    if not math.isnan(res.metrics["mi"])
                ]
            )
            / sum(
                [
                    res.num_examples
                    for _, res in results
                    if not math.isnan(res.metrics["mi"])
                ]
            )
        )

        selected_results = results

        if not self.has_triggered:
            critical_value = (
                self.high_critical_value
                if self.has_triggered
                else self.low_critical_value
            )

            lower_bound_mi = np.nanpercentile(mi, critical_value * 100)
            upper_bound_mi = np.nanpercentile(mi, (1 - critical_value) * 100)

            log(
                INFO,
                f"Lower bound MI: {lower_bound_mi}, Upper bound MI: {upper_bound_mi}",
            )

            selected_results = [
                (_, fit_res)
                for _, fit_res in results
                if lower_bound_mi <= fit_res.metrics["mi"] <= upper_bound_mi
            ]

        for _, fit_res in selected_results:
            self.client_score[int(fit_res.metrics["cid"])] += 1

        if not self.has_triggered:
            if self.trigger_round is not None:
                self.has_triggered = server_round >= self.trigger_round
            elif len(self.mi_history) > self.window_length:
                derivative = np.gradient(
                    savgol_filter(
                        self.mi_history, window_length=self.window_length, polyorder=3
                    )
                )
                self.has_triggered = (
                    len(np.where((derivative[:-1] > 0) & (derivative[1:] < 0))[0]) > 0
                )

        aggregated_ndarrays = aggregate_inplace(selected_results)

        log(INFO, f"Aggregating {len(selected_results)} fit results")
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in selected_results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy | FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        fit_configurations = []

        if self.optimize and self.has_triggered:
            population = np.array([int(c.cid) for c in clients])
            population_weights = np.take(self.client_score, population)
            population_weights = np.array(population_weights) / np.sum(
                population_weights
            )

            critical_value = (
                self.high_critical_value
                if self.has_triggered
                else self.low_critical_value
            )
            size = int(self.min_fit_clients * (1 - 2 * critical_value))
            selected_idx = np.random.choice(
                population, size=size, replace=False, p=population_weights
            )

            log(
                INFO,
                f"Selecting {size} clients: {selected_idx} from ({len(population)}) {population}",
            )

            log(
                INFO,
                f"min_fit_clients:{self.min_fit_clients} min_num_clients:{min_num_clients} sample_size:{sample_size}",
            )

            for client in clients:
                if int(client.cid) in selected_idx:
                    fit_configurations.append(
                        (client, FitIns(parameters, {"dyna_mifl": 0, **config}))
                    )
                else:
                    fit_configurations.append(
                        (client, FitIns([], {"dyna_mifl": 1, **config}))
                    )
        else:
            for client in clients:
                fit_configurations.append(
                    (client, FitIns(parameters, {"dyna_mifl": 0, **config}))
                )

        return fit_configurations
