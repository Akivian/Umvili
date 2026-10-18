"""Loss helpers shared by value-based learners."""

from __future__ import annotations

import torch


def weighted_mse(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-sample squared error, then an importance-weighted mean.

    ``weights`` must broadcast over the batch. A weight of zero drops that
    sample instead of scaling a loss that was already averaged.
    """
    per_sample = (predictions - targets).pow(2)
    if per_sample.dim() > 1:
        per_sample = per_sample.view(per_sample.size(0), -1).mean(dim=1)
    sample_weights = weights.reshape(-1).to(per_sample.dtype)
    if sample_weights.numel() != per_sample.numel():
        raise ValueError(
            f"weight count {sample_weights.numel()} != batch {per_sample.numel()}"
        )
    return (sample_weights * per_sample).mean()
