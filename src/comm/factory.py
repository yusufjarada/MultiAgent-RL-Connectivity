"""Factory for constructing communication policies."""

from typing import Optional

from torch import nn

from src.comm.commnet import CommNet
from src.comm.gated_attn import GatedAttnComm
from src.comm.ic3net import IC3Net
from src.comm.tarmac import TarMAC

COMM_METHODS = ("commnet", "ic3net", "tarmac", "gated_attn")


def build_comm_module(
    method: str,
    obs_dim: int,
    act_dim: int,
    n_agents: Optional[int] = None,
    hidden_dim: int = 64,
    msg_dim: int = 32,
    connectivity_weight: float = 0.5,
) -> nn.Module:
    """Build a parameter-shared communication policy.

    ``n_agents`` remains accepted for checkpoint metadata and compatibility,
    but no learnable layer depends on it. A constructed policy can therefore
    process any team size of at least two agents.
    """
    builders = {
        "commnet": lambda: CommNet(
            obs_dim, hidden_dim, msg_dim, act_dim, n_agents=n_agents
        ),
        "ic3net": lambda: IC3Net(
            obs_dim, hidden_dim, msg_dim, act_dim, n_agents=n_agents
        ),
        "tarmac": lambda: TarMAC(
            obs_dim, hidden_dim, msg_dim, act_dim, n_agents=n_agents, n_heads=4
        ),
        "gated_attn": lambda: GatedAttnComm(
            obs_dim,
            hidden_dim,
            msg_dim,
            act_dim,
            n_agents=n_agents,
            n_heads=4,
            connectivity_weight=connectivity_weight,
        ),
    }
    try:
        return builders[method]()
    except KeyError as exc:
        choices = ", ".join(COMM_METHODS)
        raise ValueError(
            f"unknown communication method {method!r}; choose {choices}"
        ) from exc
