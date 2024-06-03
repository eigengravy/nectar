from collections import OrderedDict
from typing import List
import torch
import flwr as fl


def get_params(model: torch.nn.ModuleList) -> List[fl.common.NDArray]:
    """Get model weights as a list of NumPy ndarrays."""
    # TODO: Handle bn_ error
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {k: torch.Tensor(v) for k, v in zip(model.state_dict().keys(), params)}
    )
    model.load_state_dict(state_dict, strict=True)
