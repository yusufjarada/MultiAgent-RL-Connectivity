"""
TarMAC (Das et al., 2019)

Targeted multi-agent communication using attention. Each agent broadcasts
a message at every timestep, but receivers use attention to weight incoming
messages by relevance rather than averaging equally.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.comm.common import validate_observations


class TarMAC(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        msg_dim: int,
        act_dim: int,
        n_agents: Optional[int] = None,
        n_heads: int = 4,
    ):
        super().__init__()
        if n_agents is not None and n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        if msg_dim % n_heads != 0:
            raise ValueError("msg_dim must be divisible by n_heads")

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.head_dim = msg_dim // n_heads

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Message generation (key and value for attention)
        self.msg_key = nn.Linear(hidden_dim, msg_dim)
        self.msg_value = nn.Linear(hidden_dim, msg_dim)

        # Query for attention (what this agent is looking for)
        self.msg_query = nn.Linear(hidden_dim, msg_dim)

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

        Returns:
            action_logits: (batch, n_agents, act_dim)
            info: dict with attention weights, messages, comm stats
        """
        B, N = validate_observations(obs, self.obs_dim)

        h = self.encoder(obs)  # (B, N, hidden_dim)

        # Generate queries, keys, values
        Q = self.msg_query(h)  # (B, N, msg_dim)
        K = self.msg_key(h)
        V = self.msg_value(h)

        # Reshape for multi-head attention: (B, n_heads, N, head_dim)
        Q = Q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, heads, N, N)

        # Mask self-attention (don't attend to own message)
        self_mask = torch.eye(N, device=obs.device).bool().unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(self_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, N, N)

        # Weighted sum of values
        attended = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)
        attended = attended.transpose(1, 2).reshape(B, N, -1)  # (B, N, msg_dim)

        # Integrate
        h = self.integrate(torch.cat([h, attended], dim=-1))

        action_logits = self.action_head(h)

        info = {
            "attn_weights": attn_weights.detach(),
            "comm_rate": 1.0,  # always broadcasting
        }
        return action_logits, info
