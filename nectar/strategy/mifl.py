from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from flwr.server.client_proxy import ClientProxy
from logging import INFO, WARNING
from flwr.common.logger import log
from flwr.common import (
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
)
import numpy as np
import math


class MIFL(FedAvg):
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
        inplace: bool = True,
        mi_type: str,
        critical_value: float,
        k_top: int = None,
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
            inplace=inplace,
        )
        # MIFL Hyperparameters
        self.mi_type = mi_type
        self.critical_value = critical_value
        self.k_top = k_top

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"MIFL(mi_type={self.mi_type}, critical_value={self.critical_value})"
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

        mi = [fit_res.metrics["mi"] for _, fit_res in results]

        lower_bound_mi = np.nanpercentile(mi, self.critical_value * 100)
        upper_bound_mi = np.nanpercentile(mi, (1 - self.critical_value) * 100)

        log(INFO, f"Lower bound MI: {lower_bound_mi}, Upper bound MI: {upper_bound_mi}")

        selected_results = [
            (_, fit_res)
            for _, fit_res in results
            if lower_bound_mi <= fit_res.metrics["mi"] <= upper_bound_mi
        ]

        if self.k_top:
            selected_results = sorted(
                selected_results, key=lambda res: res[1].metrics["mi"], reverse=True
            )[: self.k_top]

        aggregated_ndarrays = aggregate_inplace(selected_results)
        print(f"Aggregating fit results {len(selected_results)}")

        log(INFO, f"Aggregating {len(selected_results)} fit results")

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (res.num_examples, res.metrics) for _, res in selected_results
            ]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
