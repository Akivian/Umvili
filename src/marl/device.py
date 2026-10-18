"""Pick the compute device and keep inference tensors on the same device as the network."""

from __future__ import annotations

import numpy as np
import torch


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def q_values_numpy(module: torch.nn.Module, state) -> np.ndarray:
    device = next(module.parameters()).device
    tensor = torch.as_tensor(np.asarray(state, dtype=np.float32), device=device)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    with torch.no_grad():
        values = module(tensor)
    return values.detach().float().cpu().numpy()
