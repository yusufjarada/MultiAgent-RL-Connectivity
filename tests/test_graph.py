"""Quick sanity checks for graph utilities."""

import sys
sys.path.insert(0, '/Users/yusufjarada/Desktop/marl-comms')

import numpy as np
import torch
from src.utils.graph import (
    adjacency_to_laplacian,
    fiedler_value,
    connectivity_penalty_torch,
    communication_rate,
    build_adjacency_from_gates,
)


def test_complete_graph():
    # Complete graph on 4 nodes: Fiedler value = N = 4
    adj = np.ones((4, 4)) - np.eye(4)
    fv = fiedler_value(adj)
    print(f"Complete graph K4: Fiedler = {fv:.4f} (expected: 4.0)")
    assert abs(fv - 4.0) < 0.01


def test_path_graph():
    # Path graph 0-1-2-3: Fiedler value = 2 - 2*cos(pi/4) ≈ 0.586
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 0] = 1
    adj[1, 2] = adj[2, 1] = 1
    adj[2, 3] = adj[3, 2] = 1
    fv = fiedler_value(adj)
    expected = 2 - 2 * np.cos(np.pi / 4)
    print(f"Path graph P4: Fiedler = {fv:.4f} (expected: {expected:.4f})")
    assert abs(fv - expected) < 0.01


def test_disconnected_graph():
    # Two isolated pairs: Fiedler = 0
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 0] = 1
    adj[2, 3] = adj[3, 2] = 1
    fv = fiedler_value(adj)
    print(f"Disconnected graph: Fiedler = {fv:.4f} (expected: 0.0)")
    assert abs(fv) < 0.01


def test_connectivity_penalty():
    # Well-connected graph: penalty should be 0
    gate_probs = torch.ones(4, 4) * 0.9
    gate_probs.fill_diagonal_(0)
    penalty = connectivity_penalty_torch(gate_probs)
    print(f"Well-connected penalty: {penalty.item():.4f} (expected: ~0)")
    assert penalty.item() < 0.01

    # Sparse graph: penalty should be > 0
    gate_probs = torch.zeros(4, 4)
    gate_probs[0, 1] = gate_probs[1, 0] = 0.01
    penalty = connectivity_penalty_torch(gate_probs)
    print(f"Sparse graph penalty: {penalty.item():.4f} (expected: >0)")
    assert penalty.item() > 0


def test_comm_rate():
    gates = np.array([1, 1, 0, 0, 1, 0])
    rate = communication_rate(gates)
    print(f"Comm rate: {rate:.4f} (expected: 0.5)")
    assert abs(rate - 0.5) < 0.01


def test_build_adjacency():
    gates = np.array([1, 0, 1])
    adj = build_adjacency_from_gates(gates, 3)
    # Agent 0 and 2 transmit, so edges 0-1, 0-2, 2-0, 2-1 should exist
    assert adj[0, 2] == 1 and adj[2, 0] == 1
    assert adj[1, 0] == 1  # agent 1 receives from 0
    assert adj[0, 1] == 1  # agent 0 broadcasts to 1
    print(f"Adjacency from gates: OK")


if __name__ == '__main__':
    test_complete_graph()
    test_path_graph()
    test_disconnected_graph()
    test_connectivity_penalty()
    test_comm_rate()
    test_build_adjacency()
    print("\nAll tests passed.")
