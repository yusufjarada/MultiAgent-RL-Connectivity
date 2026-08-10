"""
IC3Net (Singh et al., 2019)

Gated communication. Each agent has a binary gate that controls whether
it broadcasts its message. When the gate is off, the agent stays silent.
Uses a sigmoid gate during training (differentiable) and hard threshold at test time.
"""

from typing import Optional

import torch
import torch.nn as nn

from src.comm.common import validate_observations


class IC3Net(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        msg_dim: int,
        act_dim: int,
        n_agents: Optional[int] = None,
    ):
        super().__init__()
        if n_agents is not None and n_agents < 2:
            raise ValueError("n_agents must be at least 2")

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Gate: decides whether to communicate
        self.gate_fn = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Message generation
        self.msg_fn = nn.Linear(hidden_dim, msg_dim)

        # Integration
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
            hard_gate: if True, use hard 0/1 decisions instead of soft probabilities.

        Returns:
            action_logits: (batch, n_agents, act_dim)
            info: dict with gates, messages, comm stats
        """
        _, N = validate_observations(obs, self.obs_dim)
        h = self.encoder(obs)  # (B, N, hidden_dim)

        # Compute gate probabilities
        gate_probs = self.gate_fn(h).squeeze(-1)  # (B, N)

        if hard_gate:
            gates = (gate_probs > 0.5).float()
        else:
            # Straight-through estimator: hard forward, soft backward
            gates_hard = (gate_probs > 0.5).float()
            gates = gate_probs + (gates_hard - gate_probs).detach()

        # Generate messages, masked by gate
        messages = self.msg_fn(h)  # (B, N, msg_dim)
        gated_messages = messages * gates.unsqueeze(-1)  # zero out silenced agents

        # Mean-pool received messages (from agents that are transmitting)
        msg_sum = (
            gated_messages.sum(dim=1, keepdim=True) - gated_messages
        )  # (B, N, msg_dim)
        active_count = (
            gates.sum(dim=1, keepdim=True) - gates
        )  # how many others are sending
        active_count = active_count.unsqueeze(-1).clamp(min=1.0)
        msg_mean = msg_sum / active_count

        # Integrate
        h = self.integrate(torch.cat([h, msg_mean], dim=-1))

        action_logits = self.action_head(h)

        info = {
            "messages": messages.detach(),
            "gates": gates.detach(),
            "gate_probs": gate_probs.detach(),
            "comm_rate": gates.mean().item(),
        }
        return action_logits, info
