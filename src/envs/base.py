"""Common interface for tensor-based multi-agent environments."""

from typing import Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class MultiAgentEnv(Protocol):
    """Minimal environment contract consumed by :class:`PPOTrainer`."""

    n_agents: int
    obs_dim: int
    act_dim: int
    max_steps: int

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Start an episode and return ``(n_agents, obs_dim)`` observations."""

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """Advance one step and return observations, shared reward, and done."""

    def close(self) -> None:
        """Release simulator resources."""


def validate_n_agents(n_agents: int) -> None:
    """Validate a team size shared by all supported environments."""
    if not isinstance(n_agents, int) or isinstance(n_agents, bool):
        raise TypeError("n_agents must be an integer")
    if n_agents < 2:
        raise ValueError("n_agents must be at least 2")


def validate_actions(actions: torch.Tensor, n_agents: int, act_dim: int) -> None:
    """Validate a vector of discrete per-agent actions."""
    if actions.shape != (n_agents,):
        raise ValueError(
            f"actions must have shape ({n_agents},), got {tuple(actions.shape)}"
        )
    if torch.is_floating_point(actions):
        raise TypeError("actions must contain integer action indices")
    if torch.any(actions < 0) or torch.any(actions >= act_dim):
        raise ValueError(f"actions must be in [0, {act_dim - 1}]")
