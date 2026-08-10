"""Versioned, atomic checkpoint utilities."""

import os
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import torch
from torch import nn
from torch.optim import Optimizer

CHECKPOINT_VERSION = 1


def save_training_checkpoint(
    path: Union[str, Path],
    actor: nn.Module,
    critic: nn.Module,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    metadata: Mapping[str, Any],
) -> None:
    """Atomically save all state required to evaluate or resume a run."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "metadata": dict(metadata),
        "actor_state_dict": actor.state_dict(),
        "critic_state_dict": critic.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
    }
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def load_actor_state(
    path: Union[str, Path], map_location: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """Load actor weights from a versioned or legacy actor-only checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(payload, dict) and "actor_state_dict" in payload:
        return payload["actor_state_dict"]
    return payload
