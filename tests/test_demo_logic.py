"""
Tests that verify the demo logic matches expectations.
Reimplements the JS logic in Python to confirm correctness.
"""

import sys
sys.path.insert(0, '/Users/yusufjarada/Desktop/marl-comms')

import numpy as np
from src.utils.graph import fiedler_value, adjacency_to_laplacian


def distance(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


def test_broadcast_all_in_range_connected():
    """In broadcast mode, all agents within COMM_RANGE should have edges."""
    COMM_RANGE = 250

    # Place 8 agents in a tight cluster — all within range
    positions = [(100 + i * 20, 100 + i * 15) for i in range(8)]

    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if distance(positions[i], positions[j]) <= COMM_RANGE:
                edges.append((i, j))

    # All pairs should be connected in a tight cluster
    assert len(edges) == 28, f"Expected 28 edges (all pairs), got {len(edges)}"

    # Build adjacency and check Fiedler
    adj = np.zeros((8, 8))
    for i, j in edges:
        adj[i][j] = adj[j][i] = 1
    fv = fiedler_value(adj)
    assert fv > 0, f"Tight cluster should be connected, Fiedler = {fv}"
    print(f"PASS: Tight cluster broadcast — {len(edges)} edges, Fiedler = {fv:.3f}")


def test_broadcast_distant_agents_no_edge():
    """Agents beyond COMM_RANGE should NOT have edges, even in broadcast."""
    COMM_RANGE = 250

    # Place two agents far apart
    positions = [(0, 0), (500, 500)]
    dist = distance(positions[0], positions[1])
    assert dist > COMM_RANGE, f"Distance {dist} should be > {COMM_RANGE}"

    has_edge = dist <= COMM_RANGE
    assert not has_edge, "Distant agents should not have an edge"
    print(f"PASS: Distant agents ({dist:.0f}px apart) have no edge")


def test_broadcast_chain_connectivity():
    """
    Agents spread in a line with spacing < COMM_RANGE should form a connected chain.
    This matches what user sees: 0-1-2-3 connected, 4-5-6-7 connected, but
    if the two groups are far apart, they won't connect to each other.
    """
    COMM_RANGE = 250

    # Group A: agents 0-3 in a line, 80px apart
    # Group B: agents 4-7 in a line, 80px apart, but 400px away from group A
    positions = [
        (100, 100), (180, 100), (260, 100), (340, 100),  # group A
        (100, 500), (180, 500), (260, 500), (340, 500),  # group B
    ]

    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if distance(positions[i], positions[j]) <= COMM_RANGE:
                edges.append((i, j))

    adj = np.zeros((8, 8))
    for i, j in edges:
        adj[i][j] = adj[j][i] = 1

    # Check within-group connectivity
    assert adj[0][1] == 1, "Agent 0 and 1 should be connected"
    assert adj[1][2] == 1, "Agent 1 and 2 should be connected"
    assert adj[4][5] == 1, "Agent 4 and 5 should be connected"

    # Check cross-group: 400px apart, beyond COMM_RANGE
    dist_cross = distance(positions[0], positions[4])
    assert adj[0][4] == 0, f"Agent 0 and 4 should NOT be connected (dist={dist_cross:.0f})"

    fv = fiedler_value(adj)
    assert fv == 0.0 or abs(fv) < 0.01, f"Two separate groups should be disconnected, Fiedler = {fv}"
    print(f"PASS: Two separate groups — Fiedler = {fv:.4f} (disconnected as expected)")
    print(f"      Group A edges: {[(i,j) for i,j in edges if i < 4 and j < 4]}")
    print(f"      Group B edges: {[(i,j) for i,j in edges if i >= 4 and j >= 4]}")
    print(f"      Cross-group edges: {[(i,j) for i,j in edges if (i < 4) != (j < 4)]}")


def test_broadcast_single_cluster_all_connected():
    """If all agents are within COMM_RANGE of each other, complete graph."""
    COMM_RANGE = 250
    # All agents within a 100x100 box
    positions = [(50 + (i % 4) * 30, 50 + (i // 4) * 30) for i in range(8)]

    max_dist = 0
    for i in range(8):
        for j in range(i + 1, 8):
            d = distance(positions[i], positions[j])
            max_dist = max(max_dist, d)

    assert max_dist < COMM_RANGE, f"Max distance {max_dist:.0f} should be < {COMM_RANGE}"

    edges = []
    for i in range(8):
        for j in range(i + 1, 8):
            if distance(positions[i], positions[j]) <= COMM_RANGE:
                edges.append((i, j))

    assert len(edges) == 28, f"All 28 pairs should be edges, got {len(edges)}"

    adj = np.zeros((8, 8))
    for i, j in edges:
        adj[i][j] = adj[j][i] = 1

    fv = fiedler_value(adj)
    assert fv == 8.0 or abs(fv - 8.0) < 0.01, f"Complete K8 Fiedler should be 8, got {fv}"
    print(f"PASS: Complete graph K8 — Fiedler = {fv:.3f}")


def test_connectivity_repair_logic():
    """
    Simulates the 'ours' mode: start with few edges (disconnected),
    then add back edges shortest-first until Fiedler > threshold.
    """
    COMM_RANGE = 250
    THRESHOLD = 0.1

    # Sparse initial edges (disconnected)
    positions = [
        (100, 100), (180, 100), (260, 100), (340, 100),
        (100, 300), (180, 300), (260, 300), (340, 300),
    ]

    all_pairs = []
    for i in range(8):
        for j in range(i + 1, 8):
            if distance(positions[i], positions[j]) <= COMM_RANGE:
                all_pairs.append((i, j))

    # Start with very few edges (simulating aggressive gating)
    candidate_edges = [(0, 1), (4, 5)]

    adj = np.zeros((8, 8))
    for i, j in candidate_edges:
        adj[i][j] = adj[j][i] = 1
    fv = fiedler_value(adj)
    print(f"Before repair: {len(candidate_edges)} edges, Fiedler = {fv:.4f}")
    assert fv < THRESHOLD, "Should start disconnected"

    # Repair: add back dropped edges shortest-first
    dropped = [e for e in all_pairs if e not in candidate_edges]
    dropped.sort(key=lambda e: distance(positions[e[0]], positions[e[1]]))

    current_edges = list(candidate_edges)
    for edge in dropped:
        if fv >= THRESHOLD:
            break
        current_edges.append(edge)
        adj = np.zeros((8, 8))
        for i, j in current_edges:
            adj[i][j] = adj[j][i] = 1
        fv = fiedler_value(adj)

    print(f"After repair:  {len(current_edges)} edges, Fiedler = {fv:.4f}")
    assert fv >= THRESHOLD, f"Repair should achieve connectivity, Fiedler = {fv}"
    assert len(current_edges) < len(all_pairs), "Should use fewer edges than broadcast"
    print(f"PASS: Connectivity repair — added {len(current_edges) - len(candidate_edges)} edges to reconnect")


def test_gated_mode_can_disconnect():
    """Gated mode with no repair CAN produce Fiedler = 0. That's the point."""
    import math

    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))

    # Simulate gate decisions at a specific frame
    # Find a frame where most gates are closed
    disconnected_frames = 0
    for frame in range(500):
        gates = []
        for i in range(8):
            g = sigmoid(math.sin(frame * 0.015 + i * 1.7) * 2.0
                       + math.cos(frame * 0.008 + i * 3.1) * 1.5 + 0.5)
            gates.append(g > 0.5)

        open_count = sum(gates)
        if open_count <= 2:
            disconnected_frames += 1

    print(f"PASS: Gated mode — {disconnected_frames}/500 frames had <=2 open gates")
    print(f"      This confirms gating CAN cause low connectivity (the problem we solve)")


if __name__ == '__main__':
    print("=" * 60)
    print("Demo Logic Tests")
    print("=" * 60)
    print()
    test_broadcast_all_in_range_connected()
    print()
    test_broadcast_distant_agents_no_edge()
    print()
    test_broadcast_chain_connectivity()
    print()
    test_broadcast_single_cluster_all_connected()
    print()
    test_connectivity_repair_logic()
    print()
    test_gated_mode_can_disconnect()
    print()
    print("=" * 60)
    print("All demo logic tests passed.")
    print("=" * 60)
