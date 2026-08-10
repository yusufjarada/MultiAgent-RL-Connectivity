"""Tests for variable-size communication policies."""

import pytest
import torch

from src.comm.factory import COMM_METHODS, build_comm_module


@pytest.mark.parametrize("method", COMM_METHODS)
@pytest.mark.parametrize("n_agents", (2, 5, 9))
def test_policy_supports_runtime_team_sizes(method, n_agents):
    model = build_comm_module(
        method, obs_dim=16, act_dim=5, n_agents=3, hidden_dim=32, msg_dim=16
    )
    observations = torch.randn(4, n_agents, 16)

    logits, info = model(observations, hard_gate=False)

    assert logits.shape == (4, n_agents, 5)
    assert torch.isfinite(logits).all()
    assert 0.0 <= info["comm_rate"] <= 1.0


def test_commnet_messages_and_rate():
    model = build_comm_module("commnet", obs_dim=8, act_dim=5, msg_dim=16)
    logits, info = model(torch.randn(2, 6, 8))

    assert logits.shape == (2, 6, 5)
    assert info["messages"].shape == (2, 6, 16)
    assert info["comm_rate"] == 1.0


def test_ic3net_hard_gates_are_binary():
    model = build_comm_module("ic3net", obs_dim=8, act_dim=5, msg_dim=16)
    _, info = model(torch.randn(2, 6, 8), hard_gate=True)

    assert info["gates"].shape == (2, 6)
    assert torch.all((info["gates"] == 0) | (info["gates"] == 1))


def test_tarmac_attention_excludes_self_and_normalizes():
    model = build_comm_module("tarmac", obs_dim=8, act_dim=5, msg_dim=16)
    _, info = model(torch.randn(2, 6, 8))
    attention = info["attn_weights"]

    assert attention.shape == (2, 4, 6, 6)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2, 4, 6), atol=1e-6)
    diagonal = torch.diagonal(attention, dim1=-2, dim2=-1)
    assert torch.count_nonzero(diagonal) == 0


def test_gated_attention_reports_non_self_rate_and_fiedler():
    model = build_comm_module("gated_attn", obs_dim=8, act_dim=5, msg_dim=16)
    _, info = model(torch.randn(2, 6, 8), hard_gate=True)

    assert info["gates"].shape == (2, 6, 6)
    assert info["fiedler"].shape == (2,)
    assert torch.isfinite(info["attn_weights"]).all()
    diagonal = torch.diagonal(info["gates"], dim1=-2, dim2=-1)
    assert torch.count_nonzero(diagonal) == 0

    expected_rate = info["gates"].sum().item() / (2 * 6 * 5)
    assert info["comm_rate"] == pytest.approx(expected_rate)


@pytest.mark.parametrize("method", COMM_METHODS)
def test_policy_rejects_invalid_observations(method):
    model = build_comm_module(method, obs_dim=8, act_dim=5, msg_dim=16)

    with pytest.raises(ValueError, match="at least two agents"):
        model(torch.randn(2, 1, 8))
    with pytest.raises(ValueError, match="observation dimension"):
        model(torch.randn(2, 3, 7))
