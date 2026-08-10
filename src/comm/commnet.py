"""
CommNet (Sukhbaatar et al., 2016)

Continuous communication with mean-pooling. Every agent broadcasts a message
at every timestep. Each agent averages all received messages and concatenates
with its own hidden state before acting.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.comm.common import validate_observations


class CommNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        msg_dim: int,
        act_dim: int,
        n_agents: Optional[int] = None,
        n_comm_rounds: int = 1,
    ):
        super().__init__()
        if n_agents is not None and n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        if n_comm_rounds < 1:
            raise ValueError("n_comm_rounds must be at least 1")

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_comm_rounds = n_comm_rounds
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Communication: maps hidden state to message
        self.msg_fn = nn.Linear(hidden_dim, msg_dim)

        # Integration: combines hidden state with received message
        self.integrate = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(
        self, obs: torch.Tensor, hard_gate: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            obs: (batch, n_agents, obs_dim)

        Returns:
            action_logits: (batch, n_agents, act_dim)
            info: dict with messages and comm stats
        """
        _, N = validate_observations(obs, self.obs_dim)
        h = self.encoder(obs)  # (B, N, hidden_dim)

        for _ in range(self.n_comm_rounds):
            # Every agent broadcasts
            messages = self.msg_fn(h)  # (B, N, msg_dim)

            # Mean-pool all other agents' messages
            msg_sum = messages.sum(dim=1, keepdim=True) - messages  # (B, N, msg_dim)
            msg_mean = msg_sum / (N - 1)

            # Integrate
            h = self.integrate(torch.cat([h, msg_mean], dim=-1))

        action_logits = self.action_head(h)

        info = {
            "messages": messages.detach(),
            "comm_rate": 1.0,  # always broadcasting
        }
        return action_logits, info
