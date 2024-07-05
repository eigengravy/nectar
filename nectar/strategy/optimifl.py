import random
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from logging import INFO, WARNING
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
from sklearn.linear_model import LinearRegression
from nectar.strategy.mifl import MIFL


class OptiMIFL(MIFL):
    def __init__(
        self,
        *,
        fraction_fit: float = 0.8,
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
        opti_rounds: int = 15,
        lottery_ratio: float = 0.25,
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
            mi_type=mi_type,
            critical_value=critical_value,
        )
        self.lottery_ratio = lottery_ratio
        self.opti_rounds = opti_rounds
        self.lower_mi_history = []
        self.upper_mi_history = []
        self.lr_lower = LinearRegression()
        self.lr_upper = LinearRegression()

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"OptiMIFL(mi_type={self.mi_type}, critical_value={self.critical_value})"
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

        log(INFO, f"Received {len(results)} fit results")

        results = [result for result in results if result[1].metrics["cid"] != -1]

        mi = [
            fit_res.metrics["mi"]
            for _, fit_res in results
            if not np.isnan(fit_res.metrics["mi"])
        ]
        log(INFO, f"len(mi)={len(mi)}")

        if len(mi) < self.min_fit_clients:
            mi.extend([sum(mi) / len(mi)] * (self.min_fit_clients - len(mi)))
        log(INFO, f"len(mi)={len(mi)}")

        # if server_round < self.opti_rounds:
        lower_bound_mi = np.percentile(mi, self.critical_value * 100)
        upper_bound_mi = np.percentile(mi, (1 - self.critical_value) * 100)
        # else:
        #     X = np.array([self.lower_mi_history[-2], self.upper_mi_history[-2]]).reshape(1, -1)
        #     lower_bound_mi = self.lr_lower.predict(X.reshape(1, -1))[0]
        #     upper_bound_mi = self.lr_upper.predict(X.reshape(1, -1))[0]

        log(INFO, f"Lower bound: {lower_bound_mi}, Upper bound: {upper_bound_mi}")
        self.lower_mi_history.append(lower_bound_mi)
        self.upper_mi_history.append(upper_bound_mi)

        if self.inplace:
            # Does in-place weighted average of results
            # selected_results = [
            #     (_, fit_res)
            #     for _, fit_res in results
            #     if lower_bound_mi <= fit_res.metrics["mi"] <= upper_bound_mi
            # ]
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            # selected_results = [
            #     (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            #     for _, fit_res in results
            #     if lower_bound_mi <= fit_res.metrics["mi"] <= upper_bound_mi
            # ]
            aggregated_ndarrays = aggregate(results)

        log(INFO, f"Aggregating {len(results)} fit results")

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
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
            num_clients=sample_size, min_num_clients=sample_size
        )

        lottery_idx = random.sample(
            range(len(clients)), int(len(clients) * self.lottery_ratio)
        )

        log(INFO, f"Lottery: {lottery_idx}")

        if server_round == self.opti_rounds:
            # Fit on entire history
            log(INFO, "Fit entire history")
            X = np.array(list(zip(self.lower_mi_history, self.upper_mi_history))[:-1])
            y_lower = np.array(self.lower_mi_history[1:])
            y_upper = np.array(self.upper_mi_history[1:])
            self.lr_lower.fit(X, y_lower)
            self.lr_upper.fit(X, y_upper)
            lower_bound = self.lr_lower.predict(X[-1].reshape(1, -1))[0]
            upper_bound = self.lr_upper.predict(X[-1].reshape(1, -1))[0]
            log(
                INFO,
                f"Lower bound MI (Predicted): {lower_bound}, Upper bound MI (Predicted): {upper_bound}",
            )
        elif server_round > self.opti_rounds:
            # Fit on last round
            log(INFO, "Fit last round")
            X = np.array(
                [self.lower_mi_history[-2], self.upper_mi_history[-2]]
            ).reshape(1, -1)
            self.lr_lower.fit(X, np.array(self.lower_mi_history[-1]).reshape(-1, 1))
            self.lr_upper.fit(X, np.array(self.upper_mi_history[-1]).reshape(-1, 1))
            lower_bound = self.lr_lower.predict(X.reshape(1, -1))[0][0]
            upper_bound = self.lr_upper.predict(X.reshape(1, -1))[0][0]
            log(
                INFO,
                f"Lower bound MI (Predicted): {lower_bound}, Upper bound MI (Predicted): {upper_bound}",
            )

        fit_configurations = []
        if server_round < self.opti_rounds:
            for idx, client in enumerate(clients):
                fit_configurations.append(
                    (client, FitIns(parameters, {"opti_mifl": 0, **config}))
                )
        else:
            for idx, client in enumerate(clients):
                if idx in lottery_idx:
                    fit_configurations.append(
                        (client, FitIns(parameters, {"opti_mifl": 0, **config}))
                    )
                else:
                    fit_configurations.append(
                        (
                            client,
                            FitIns(
                                parameters,
                                {
                                    **config,
                                    "opti_mifl": 1,
                                    "lower_bound": lower_bound,
                                    "upper_bound": upper_bound,
                                },
                            ),
                        )
                    )
        return fit_configurations
