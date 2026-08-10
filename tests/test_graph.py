"""Tests for graph utilities and differentiable connectivity metrics."""

import numpy as np
import pytest
import torch

from src.utils.graph import (
    adjacency_to_laplacian,
    algebraic_connectivity_torch,
    build_adjacency_from_gates,
    communication_rate,
    connectivity_penalty_torch,
    fiedler_value,
)


def test_complete_graph():
    adjacency = np.ones((4, 4)) - np.eye(4)
    assert fiedler_value(adjacency) == pytest.approx(4.0)


def test_path_graph():
    adjacency = np.zeros((4, 4))
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[1, 2] = adjacency[2, 1] = 1
    adjacency[2, 3] = adjacency[3, 2] = 1
    expected = 2 - 2 * np.cos(np.pi / 4)
    assert fiedler_value(adjacency) == pytest.approx(expected)


def test_disconnected_graph():
    adjacency = np.zeros((4, 4))
    adjacency[0, 1] = adjacency[1, 0] = 1
    adjacency[2, 3] = adjacency[3, 2] = 1
    assert abs(fiedler_value(adjacency)) < 1e-7


def test_batched_algebraic_connectivity():
    complete = torch.ones(5, 5) - torch.eye(5)
    disconnected = torch.zeros(5, 5)
    values = algebraic_connectivity_torch(torch.stack((complete, disconnected)))

    assert values.shape == (2,)
    assert values[0].item() == pytest.approx(5.0, abs=1e-5)
    assert abs(values[1].item()) < 1e-7


def test_connectivity_penalty_is_differentiable():
    logits = torch.randn(4, 4, requires_grad=True)
    diagonal_mask = 1.0 - torch.eye(4)
    sparse_probabilities = torch.sigmoid(logits - 6.0) * diagonal_mask

    penalty = connectivity_penalty_torch(sparse_probabilities)
    penalty.backward()

    assert penalty.item() > 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0


def test_range_limited_penalty():
    probabilities = torch.ones(3, 3) - torch.eye(3)
    near = torch.tensor(((0.0, 0.0), (0.5, 0.0), (1.0, 0.0)))
    far = torch.tensor(((0.0, 0.0), (0.5, 0.0), (5.0, 0.0)))

    assert connectivity_penalty_torch(
        probabilities, positions=near, comm_range=1.1
    ).item() == pytest.approx(0.0)
    assert (
        connectivity_penalty_torch(probabilities, positions=far, comm_range=1.1).item()
        > 0
    )


def test_graph_input_validation():
    with pytest.raises(ValueError, match="square"):
        adjacency_to_laplacian(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="at least two"):
        fiedler_value(np.zeros((1, 1)))
    with pytest.raises(ValueError, match="provided together"):
        connectivity_penalty_torch(torch.ones(2, 2), comm_range=1.0)


def test_legacy_gate_helpers():
    gates = np.array([1, 1, 0, 0, 1, 0])
    assert communication_rate(gates) == pytest.approx(0.5)

    adjacency = build_adjacency_from_gates(np.array([1, 0, 1]), 3)
    assert np.array_equal(adjacency, adjacency.T)
    assert np.count_nonzero(np.diag(adjacency)) == 0
