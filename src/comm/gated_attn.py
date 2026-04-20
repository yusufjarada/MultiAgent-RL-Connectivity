"""
Gated Attention Communication with Connectivity Constraint (Ours)

Combines IC3Net-style gating with TarMAC-style attention targeting.
Adds a differentiable algebraic connectivity penalty to prevent the
learned gates from disconnecting the communication graph.

This is the novel contribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.utils.graph import connectivity_penalty_torch


class GatedAttnComm(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, msg_dim: int, act_dim: int,
                 n_agents: int, n_heads: int = 4, connectivity_weight: float = 1.0,
                 gate_temp: float = 1.0):
        super().__init__()
        self.n_agents = n_agents
        self.n_heads = n_heads
        self.head_dim = msg_dim // n_heads
        self.connectivity_weight = connectivity_weight
        self.gate_temp = gate_temp
        assert msg_dim % n_heads == 0

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

    def _compute_gates(self, h: torch.Tensor, hard: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Build all pairs: (B, N, N, 2*hidden_dim)
        h_sender = h.unsqueeze(2).expand(B, N, N, D)  # (B, N, N, D)
        h_receiver = h.unsqueeze(1).expand(B, N, N, D)  # (B, N, N, D)
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

    def forward(self, obs: torch.Tensor, hard_gate: bool = False,
                agent_positions: torch.Tensor = None) -> tuple[torch.Tensor, dict]:
        """
        Args:
            obs: (batch, n_agents, obs_dim)
            hard_gate: if True, use hard 0/1 gate decisions
            agent_positions: (batch, n_agents, 2) optional positions for range-limited comm

        Returns:
            action_logits: (batch, n_agents, act_dim)
            info: dict with gates, attention, connectivity penalty, comm stats
        """
        B, N, _ = obs.shape

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
        attn_scores = attn_scores.masked_fill(gate_mask == 0, float('-inf'))

        # Also mask self-attention
        self_mask = torch.eye(N, device=obs.device).bool().unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(self_mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, heads, N, N)
        # Handle case where all connections are gated off (softmax gives nan)
        attn_weights = attn_weights.nan_to_num(0.0)

        # Weighted sum of values
        attended = torch.matmul(attn_weights, V)  # (B, heads, N, head_dim)
        attended = attended.transpose(1, 2).reshape(B, N, -1)  # (B, N, msg_dim)

        # Integrate
        h = self.integrate(torch.cat([h, attended], dim=-1))
        action_logits = self.action_head(h)

        # Connectivity penalty (averaged over batch)
        conn_penalty = torch.tensor(0.0, device=obs.device)
        for b in range(B):
            conn_penalty = conn_penalty + connectivity_penalty_torch(gate_probs[b])
        conn_penalty = conn_penalty / B * self.connectivity_weight

        info = {
            "gates": gates.detach(),
            "gate_probs": gate_probs.detach(),
            "attn_weights": attn_weights.detach(),
            "conn_penalty": conn_penalty,
            "fiedler_approx": -conn_penalty.item() + 0.1,  # rough estimate
            "comm_rate": gates.mean().item(),
        }
        return action_logits, info
