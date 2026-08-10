"""Shared validation and metrics for communication policies."""

from typing import Tuple

import torch


def validate_observations(obs: torch.Tensor, obs_dim: int) -> Tuple[int, int]:
    """Validate a batched multi-agent observation tensor.

    Communication policies share parameters across agents, so the team size is
    intentionally inferred from each input rather than fixed by the model.
    """
    if obs.ndim != 3:
        raise ValueError(
            "obs must have shape (batch, n_agents, obs_dim); "
            f"received {tuple(obs.shape)}"
        )

    batch_size, n_agents, actual_obs_dim = obs.shape
    if n_agents < 2:
        raise ValueError("communication policies require at least two agents")
    if actual_obs_dim != obs_dim:
        raise ValueError(
            f"expected observation dimension {obs_dim}, got {actual_obs_dim}"
        )
    return batch_size, n_agents


def pairwise_communication_rate(gates: torch.Tensor) -> float:
    """Return the fraction of active directed, non-self communication links."""
    if gates.ndim != 3 or gates.shape[-1] != gates.shape[-2]:
        raise ValueError("gates must have shape (batch, n_agents, n_agents)")

    n_agents = gates.shape[-1]
    if n_agents < 2:
        raise ValueError("communication rates require at least two agents")

    diagonal = torch.diagonal(gates, dim1=-2, dim2=-1).sum()
    active_links = gates.sum() - diagonal
    possible_links = gates.shape[0] * n_agents * (n_agents - 1)
    return float((active_links / possible_links).item())
