from collections import OrderedDict
from typing import List
import torch
import flwr as fl


def get_params(model: torch.nn.ModuleList) -> List[fl.common.NDArray]:
    """Get model parameters as a list of NumPy ndarrays.

    Args:
    model (torch.nn.ModuleList): Model

    Returns:
    List[fl.common.NDArray]: List of NumPy ndarrays
    """
    return [
        val.cpu().numpy()
        for name, val in model.state_dict().items()
        if "bn" not in name
    ]


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]) -> None:
    """Set model weights from a list of NumPy ndarrays.

    Args:
    model (torch.nn.ModuleList): Model
    params (List[fl.common.NDArray]): List of NumPy ndarrays

    Returns:
    None
    """
    params_dict = zip([k for k in model.state_dict().keys() if "bn" not in k], params)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)
