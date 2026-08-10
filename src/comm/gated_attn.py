"""
Gated Attention Communication with Connectivity Constraint (Ours)

Combines IC3Net-style gating with TarMAC-style attention targeting.
Adds a differentiable algebraic connectivity penalty to prevent the
learned gates from disconnecting the communication graph.

This is the novel contribution.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.comm.common import pairwise_communication_rate, validate_observations
from src.utils.graph import (
    DEFAULT_CONNECTIVITY_THRESHOLD,
    algebraic_connectivity_torch,
)


class GatedAttnComm(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        msg_dim: int,
        act_dim: int,
        n_agents: Optional[int] = None,
        n_heads: int = 4,
        connectivity_weight: float = 1.0,
        gate_temp: float = 1.0,
        connectivity_threshold: float = DEFAULT_CONNECTIVITY_THRESHOLD,
    ):
        super().__init__()
        if n_agents is not None and n_agents < 2:
            raise ValueError("n_agents must be at least 2")
        if msg_dim % n_heads != 0:
            raise ValueError("msg_dim must be divisible by n_heads")
        if gate_temp <= 0:
            raise ValueError("gate_temp must be positive")
        if connectivity_weight < 0:
            raise ValueError("connectivity_weight must be non-negative")
        if connectivity_threshold < 0:
            raise ValueError("connectivity_threshold must be non-negative")

        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.head_dim = msg_dim // n_heads
        self.connectivity_weight = connectivity_weight
        self.connectivity_threshold = connectivity_threshold
        self.gate_temp = gate_temp

        # Observation encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        # Pairwise gate: should agent i send to agent j?
        # Takes concatenation of sender and receiver embeddings
        self.gate_fn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Message generation (key, value for attention)
        self.msg_key = nn.Linear(hidden_dim, msg_dim)
        self.msg_value = nn.Linear(hidden_dim, msg_dim)
        self.msg_query = nn.Linear(hidden_dim, msg_dim)

        # Integration
        self.integrate = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, hidden_dim),
            nn.ReLU(),
        )

        # Action head
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def _compute_gates(
        self, h: torch.Tensor, hard: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute pairwise communication gates.

        Args:
            h: (B, N, hidden_dim) agent embeddings
            hard: use hard 0/1 decisions

        Returns:
            gates: (B, N, N) gate decisions (0 or 1, or soft probabilities)
            gate_probs: (B, N, N) raw sigmoid probabilities
        """
        B, N, D = h.shape

        # Matrix entry [receiver, sender] controls an incoming message.
        h_receiver = h.unsqueeze(2).expand(B, N, N, D)
        h_sender = h.unsqueeze(1).expand(B, N, N, D)
        pairs = torch.cat([h_sender, h_receiver], dim=-1)  # (B, N, N, 2D)

        gate_logits = self.gate_fn(pairs).squeeze(-1)  # (B, N, N)
        gate_probs = torch.sigmoid(gate_logits / self.gate_temp)

        # Zero out self-connections
        self_mask = torch.eye(N, device=h.device).bool().unsqueeze(0)
        gate_probs = gate_probs.masked_fill(self_mask, 0.0)

        if hard:
            gates = (gate_probs > 0.5).float()
        else:
            # Straight-through estimator
            gates_hard = (gate_probs > 0.5).float()
            gates = gate_probs + (gates_hard - gate_probs).detach()

        return gates, gate_probs

    def forward(
        self,
        obs: torch.Tensor,
        hard_gate: bool = False,
        agent_positions: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            obs: (batch, n_agents, obs_dim)
            hard_gate: if True, use hard 0/1 gate decisions
            agent_positions: (batch, n_agents, 2) optional positions for range-limited comm

        Returns:
            action_logits: (batch, n_agents, act_dim)
            info: dict with gates, attention, connectivity penalty, comm stats
        """
        B, N = validate_observations(obs, self.obs_dim)

        h = self.encoder(obs)  # (B, N, hidden_dim)

        # Compute pairwise gates
        gates, gate_probs = self._compute_gates(h, hard=hard_gate)  # (B, N, N)

        # Generate queries, keys, values
        Q = self.msg_query(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.msg_key(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.msg_value(h).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, heads, N, N)

        # Apply gate mask: if gate[i][j] = 0, agent i can't receive from j
        # Expand gates for multi-head: (B, 1, N, N)
        gate_mask = gates.unsqueeze(1)
        # Mask out gated-off connections (set to -inf before softmax)
        attn_scores = attn_scores.masked_fill(gate_mask == 0, float("-inf"))

        # Also mask self-attention
        self_mask = torch.eye(N, device=obs.device).bool().unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(self_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, N, N)
        # Handle case where all connections are gated off (softmax gives nan)
        attn_weights = attn_weights.nan_to_num(0.0)

        # Weighted sum of values
        attended = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)
        attended = attended.transpose(1, 2).reshape(B, N, -1)  # (B, N, msg_dim)

        # Integrate
        h = self.integrate(torch.cat([h, attended], dim=-1))
        action_logits = self.action_head(h)

        fiedler_values = algebraic_connectivity_torch(gate_probs)
        raw_penalty = torch.relu(self.connectivity_threshold - fiedler_values).mean()
        conn_penalty = raw_penalty * self.connectivity_weight

        info = {
            "gates": gates.detach(),
            "gate_probs": gate_probs.detach(),
            "attn_weights": attn_weights.detach(),
            "conn_penalty": conn_penalty,
            "fiedler": fiedler_values.detach(),
            "comm_rate": pairwise_communication_rate(gates.detach()),
        }
        return action_logits, info
