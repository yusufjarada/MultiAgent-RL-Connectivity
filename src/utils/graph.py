"""
Graph theory utilities for communication topology analysis.

Core functions: Laplacian matrix, Fiedler value (algebraic connectivity),
and a differentiable connectivity penalty for use in training.
"""

import numpy as np
import torch
from scipy.linalg import eigvalsh

DEFAULT_CONNECTIVITY_THRESHOLD = 0.1


def _validate_square_matrix(matrix, name: str) -> int:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    n_agents = matrix.shape[0]
    if n_agents < 2:
        raise ValueError(f"{name} must describe at least two agents")
    return n_agents


def adjacency_to_laplacian(adj: np.ndarray) -> np.ndarray:
    """Compute the graph Laplacian L = D - A from an adjacency matrix."""
    _validate_square_matrix(adj, "adj")
    degree = np.diag(adj.sum(axis=1))
    return degree - adj


def fiedler_value(adj: np.ndarray) -> float:
    """
    Compute the Fiedler value (second-smallest eigenvalue of the Laplacian).

    If > 0, the graph is connected. If == 0, the graph is disconnected.
    Larger values mean stronger connectivity.
    """
    L = adjacency_to_laplacian(adj)
    eigenvalues = eigvalsh(L)
    # Second smallest eigenvalue (first is always 0 for connected components)
    return float(eigenvalues[1])


def fiedler_value_batch(adj_batch: np.ndarray) -> np.ndarray:
    """Compute Fiedler value for a batch of adjacency matrices. Shape: (B, N, N)."""
    batch_size = adj_batch.shape[0]
    values = np.zeros(batch_size)
    for i in range(batch_size):
        values[i] = fiedler_value(adj_batch[i])
    return values


def algebraic_connectivity_torch(gate_probs: torch.Tensor) -> torch.Tensor:
    """Compute algebraic connectivity for one or more soft directed graphs.

    Directed probabilities are symmetrized before constructing the Laplacian.
    Inputs may have shape ``(N, N)`` or ``(B, N, N)``.
    """
    if gate_probs.ndim not in (2, 3):
        raise ValueError("gate_probs must have shape (N, N) or (B, N, N)")
    if gate_probs.shape[-1] != gate_probs.shape[-2]:
        raise ValueError("gate_probs must contain square adjacency matrices")

    n_agents = gate_probs.shape[-1]
    if n_agents < 2:
        raise ValueError("gate_probs must describe at least two agents")

    adj_soft = (gate_probs + gate_probs.transpose(-1, -2)) / 2.0
    identity = torch.eye(n_agents, device=gate_probs.device, dtype=gate_probs.dtype)
    adj_soft = adj_soft * (1.0 - identity)
    degree = adj_soft.sum(dim=-1)
    laplacian = torch.diag_embed(degree) - adj_soft
    eigenvalues = torch.linalg.eigvalsh(laplacian)
    return eigenvalues[..., 1]


def connectivity_penalty_torch(
    gate_probs: torch.Tensor,
    positions: torch.Tensor = None,
    comm_range: float = None,
    threshold: float = DEFAULT_CONNECTIVITY_THRESHOLD,
) -> torch.Tensor:
    """
    Differentiable connectivity penalty based on the soft graph Laplacian.

    Uses a soft adjacency matrix built from gate probabilities to compute
    an approximate Fiedler value. Penalizes when connectivity drops.

    Args:
        gate_probs: (N, N) matrix of communication probabilities between agents.
                    gate_probs[i, j] = probability that agent i sends to agent j.
        positions: (N, 2) agent positions. If provided with comm_range, masks
                   edges beyond communication range before applying gates.
        comm_range: max distance for communication. Only used if positions given.

    Returns:
        Scalar penalty (0 when well-connected, positive when near-disconnected).
    """
    n_agents = _validate_square_matrix(gate_probs, "gate_probs")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if (positions is None) != (comm_range is None):
        raise ValueError("positions and comm_range must be provided together")

    # Build soft adjacency: symmetrize the gate probabilities
    adj_soft = (gate_probs + gate_probs.T) / 2.0

    # If positions and range given, zero out edges beyond range
    if positions is not None and comm_range is not None:
        if positions.shape != (n_agents, 2):
            raise ValueError(
                f"positions must have shape ({n_agents}, 2), got {tuple(positions.shape)}"
            )
        if comm_range <= 0:
            raise ValueError("comm_range must be positive")
        diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 2)
        dist = torch.norm(diff, dim=-1)  # (N, N)
        range_mask = (dist <= comm_range).float()
        adj_soft = adj_soft * range_mask

    # Zero diagonal
    adj_soft = adj_soft * (
        1.0 - torch.eye(n_agents, device=gate_probs.device, dtype=gate_probs.dtype)
    )
    fiedler = algebraic_connectivity_torch(adj_soft)

    # Penalty: we want fiedler > some threshold (e.g., 0.1)
    # ReLU-style: penalty kicks in when fiedler drops below threshold
    penalty = torch.relu(threshold - fiedler)

    return penalty


def communication_rate(gates: np.ndarray) -> float:
    """
    Compute communication rate = fraction of active gates.

    Args:
        gates: binary array of gate decisions, any shape.

    Returns:
        Float in [0, 1].
    """
    if gates.size == 0:
        raise ValueError("gates cannot be empty")
    return float(gates.mean())


def build_adjacency_from_gates(gates: np.ndarray, n_agents: int) -> np.ndarray:
    """
    Build an adjacency matrix from a flat or structured gate vector.

    Args:
        gates: (N,) binary vector where gates[i] = 1 means agent i is transmitting.
               When an agent transmits, all other agents receive (broadcast gating).
        n_agents: number of agents.

    Returns:
        (N, N) symmetric adjacency matrix.
    """
    adj = np.zeros((n_agents, n_agents))
    for i in range(n_agents):
        if gates[i]:
            for j in range(n_agents):
                if i != j:
                    adj[i, j] = 1
                    adj[j, i] = 1
    return adj
