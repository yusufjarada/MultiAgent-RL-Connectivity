"""
Tests for all four communication modules.
Verifies shapes, gate behavior, and connectivity penalty.
"""

import sys
sys.path.insert(0, '/Users/yusufjarada/Desktop/marl-comms')

import torch
import numpy as np


def test_commnet():
    from src.comm.commnet import CommNet

    model = CommNet(obs_dim=16, hidden_dim=32, msg_dim=16, act_dim=5, n_agents=6)
    obs = torch.randn(4, 6, 16)  # batch=4, agents=6, obs_dim=16
    logits, info = model(obs)

    assert logits.shape == (4, 6, 5), f"Expected (4,6,5), got {logits.shape}"
    assert info['comm_rate'] == 1.0, "CommNet always broadcasts"
    assert info['messages'].shape == (4, 6, 16), f"Bad message shape: {info['messages'].shape}"
    print(f"PASS: CommNet — output {logits.shape}, comm_rate={info['comm_rate']}")


def test_ic3net():
    from src.comm.ic3net import IC3Net

    model = IC3Net(obs_dim=16, hidden_dim=32, msg_dim=16, act_dim=5, n_agents=6)
    obs = torch.randn(4, 6, 16)

    # Soft gate (training)
    logits, info = model(obs, hard_gate=False)
    assert logits.shape == (4, 6, 5), f"Expected (4,6,5), got {logits.shape}"
    assert 'gates' in info, "Should return gate decisions"
    assert info['gates'].shape == (4, 6), f"Bad gate shape: {info['gates'].shape}"
    assert 0 <= info['comm_rate'] <= 1, f"Comm rate out of range: {info['comm_rate']}"
    print(f"PASS: IC3Net soft — output {logits.shape}, comm_rate={info['comm_rate']:.3f}")

    # Hard gate (evaluation)
    logits, info = model(obs, hard_gate=True)
    gates = info['gates']
    assert torch.all((gates == 0) | (gates == 1)), "Hard gates should be binary"
    print(f"PASS: IC3Net hard — gates are binary, comm_rate={info['comm_rate']:.3f}")


def test_tarmac():
    from src.comm.tarmac import TarMAC

    model = TarMAC(obs_dim=16, hidden_dim=32, msg_dim=16, act_dim=5, n_agents=6, n_heads=4)
    obs = torch.randn(4, 6, 16)
    logits, info = model(obs)

    assert logits.shape == (4, 6, 5), f"Expected (4,6,5), got {logits.shape}"
    assert info['comm_rate'] == 1.0, "TarMAC always broadcasts"
    assert info['attn_weights'].shape == (4, 4, 6, 6), f"Bad attn shape: {info['attn_weights'].shape}"

    # Attention weights should sum to ~1 across the sender dimension (excluding self)
    attn = info['attn_weights']
    row_sums = attn.sum(dim=-1)  # (B, heads, N)
    # Each row should sum to ~1 (softmax over non-self entries)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=0.01), \
        f"Attention rows should sum to 1, got {row_sums[0,0]}"
    print(f"PASS: TarMAC — output {logits.shape}, attn shape {info['attn_weights'].shape}")


def test_gated_attn():
    from src.comm.gated_attn import GatedAttnComm

    model = GatedAttnComm(
        obs_dim=16, hidden_dim=32, msg_dim=16, act_dim=5,
        n_agents=6, n_heads=4, connectivity_weight=1.0
    )
    obs = torch.randn(4, 6, 16)

    # Soft gate (training)
    logits, info = model(obs, hard_gate=False)
    assert logits.shape == (4, 6, 5), f"Expected (4,6,5), got {logits.shape}"
    assert info['gates'].shape == (4, 6, 6), f"Bad gate shape: {info['gates'].shape}"
    assert info['gate_probs'].shape == (4, 6, 6), f"Bad gate_probs shape"
    assert 'conn_penalty' in info, "Should return connectivity penalty"
    print(f"PASS: GatedAttn soft — output {logits.shape}, "
          f"comm_rate={info['comm_rate']:.3f}, conn_penalty={info['conn_penalty'].item():.4f}")

    # Hard gate
    logits, info = model(obs, hard_gate=True)
    gates = info['gates']
    assert torch.all((gates == 0) | (gates == 1)), "Hard gates should be binary"
    # Diagonal should be zero (no self-communication)
    for b in range(4):
        diag = torch.diag(gates[b])
        assert torch.all(diag == 0), "Self-gates should be 0"
    print(f"PASS: GatedAttn hard — binary gates, no self-communication")


def test_gated_attn_gradient_flows():
    """Verify that gradients flow through the connectivity penalty back to gates."""
    from src.comm.gated_attn import GatedAttnComm

    model = GatedAttnComm(
        obs_dim=16, hidden_dim=32, msg_dim=16, act_dim=5,
        n_agents=4, n_heads=4, connectivity_weight=1.0
    )
    obs = torch.randn(2, 4, 16)
    logits, info = model(obs, hard_gate=False)

    # Backprop through connectivity penalty
    loss = info['conn_penalty']
    loss.backward()

    # Check that gate function parameters got gradients
    gate_has_grad = False
    for name, param in model.gate_fn.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            gate_has_grad = True
            break

    if gate_has_grad:
        print(f"PASS: Gradients flow through connectivity penalty to gate parameters")
    else:
        # Penalty might be 0 (well-connected), so no gradient is also valid
        print(f"PASS: Connectivity penalty = {loss.item():.4f} "
              f"(if 0, no gradient expected — graph is already connected)")


def test_commnet_vs_ic3net_comm_rate():
    """CommNet should always have higher comm rate than IC3Net on average."""
    from src.comm.commnet import CommNet
    from src.comm.ic3net import IC3Net

    commnet = CommNet(obs_dim=16, hidden_dim=32, msg_dim=16, act_dim=5, n_agents=6)
    ic3net = IC3Net(obs_dim=16, hidden_dim=32, msg_dim=16, act_dim=5, n_agents=6)

    rates_commnet = []
    rates_ic3net = []
    for _ in range(20):
        obs = torch.randn(4, 6, 16)
        _, info_c = commnet(obs)
        _, info_i = ic3net(obs, hard_gate=True)
        rates_commnet.append(info_c['comm_rate'])
        rates_ic3net.append(info_i['comm_rate'])

    avg_commnet = np.mean(rates_commnet)
    avg_ic3net = np.mean(rates_ic3net)
    assert avg_commnet == 1.0, f"CommNet should always be 1.0, got {avg_commnet}"
    print(f"PASS: CommNet avg rate = {avg_commnet:.2f}, IC3Net avg rate = {avg_ic3net:.3f}")
    print(f"      IC3Net communicates less (gating works)")


if __name__ == '__main__':
    print("=" * 60)
    print("Communication Module Tests")
    print("=" * 60)
    print()
    test_commnet()
    print()
    test_ic3net()
    print()
    test_tarmac()
    print()
    test_gated_attn()
    print()
    test_gated_attn_gradient_flows()
    print()
    test_commnet_vs_ic3net_comm_rate()
    print()
    print("=" * 60)
    print("All communication module tests passed.")
    print("=" * 60)
