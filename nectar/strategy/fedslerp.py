from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.strategy import FedAvg
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
    parameters_to_ndarrays,
)
import numpy as np
import math


class FedSlerp(FedAvg):
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

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"FedSlerp"
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

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
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


def aggregate_inplace(results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
    """Compute in-place weighted average."""
    # Count total examples
    num_examples_total = sum(fit_res.num_examples for (_, fit_res) in results)

    # Compute scaling factors for each result
    scaling_factors = [
        fit_res.num_examples / fit_res.num_examples for _, fit_res in results
    ]

    return slerp_n_matrices(
        [parameters_to_ndarrays(result[1].parameters) for result in results],
        scaling_factors,
    )

def slerp_n_matrices(matrices, weights):
    result = matrices[0]
    cumulative_weight = weights[0]

    for i in range(1, len(matrices)):
        t = cumulative_weight / (cumulative_weight + weights[i])
        result = [slerp(t, v0, v1) for v0, v1 in zip(result, matrices[i])]
        cumulative_weight += weights[i]

    return result


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """
    Spherical Linear Interpolation (SLERP) between two vectors.

    Args:
        t (float/np.ndarray): Interpolation factor between 0.0 and 1.0, where 0 returns v0, and 1 returns v1.
        v0 (np.ndarray): The starting vector, from which the interpolation starts.
        v1 (np.ndarray): The destination vector, to which the interpolation goes.
        DOT_THRESHOLD (float): A threshold for dot product to handle nearly parallel vectors where SLERP simplifies to LERP.

    Returns:
        np.ndarray: The interpolated vector between v0 and v1 at the position specified by t.
    """

    # Make copies of the vectors to avoid altering the originals during normalization.
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to ensure they lie on the unit sphere. This is crucial for the geometric calculations in SLERP.
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)

    # Calculate the dot product between normalized vectors to find the cosine of the angle between them. This helps determine how 'aligned' the vectors are.
    dot = np.sum(v0 * v1)

    # If vectors are nearly parallel (dot product close to 1), interpolation simplifies to linear interpolation (LERP).
    if np.abs(dot) > DOT_THRESHOLD:
        # Directly interpolate between the original vectors without spherical adjustment.
        return lerp(t, v0_copy, v1_copy)

    # Calculate the angle between the vectors using the arc cosine of the dot product.
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)  # The sine of the angle is used in the SLERP formula.

    # Compute the actual angle for the interpolation factor 't'.
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)

    # Calculate the scale factors for each vector, based on the interpolation factor and the sine values.
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0

    # Compute the final interpolated vector as a weighted sum of the original vectors.
    v2 = s0 * v0_copy + s1 * v1_copy

    return v2


def lerp(t: float, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    """
    Performs linear interpolation between two vectors or tensors.

    Args:
        t (float): The interpolation factor, where 0.0 returns `v0`, and 1.0 returns `v1`. Values between 0.0 and 1.0
        will return a weighted average of `v0` and `v1`.
        v0 (np.ndarray): The starting vector/tensor, from which the interpolation begins.
        v1 (np.ndarray): The ending vector/tensor, to which the interpolation goes.

    Returns:
        np.ndarray: The interpolated vector/tensor between `v0` and `v1` based on the interpolation factor `t`.
    """
    return (1 - t) * v0 + t * v1
