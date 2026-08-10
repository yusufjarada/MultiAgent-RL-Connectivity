"""Integration tests for result and export script path handling."""

import json

import pytest

from scripts.export_model_to_json import _checkpoint_path
from scripts.plot_results import load_results


def _write_result(root, environment, n_agents, method="commnet"):
    path = root / environment / f"{n_agents}_agents" / f"{method}_seed0.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "method": method,
                "environment": environment,
                "n_agents": n_agents,
                "rewards": [1.0],
                "comm_rates": [1.0],
                "conn_losses": [0.0],
            }
        )
    )


def test_plot_loader_filters_namespaced_results(tmp_path):
    _write_result(tmp_path, "mpe", 3)
    _write_result(tmp_path, "mujoco", 5, method="gated_attn")

    with pytest.raises(ValueError, match="multiple"):
        load_results(str(tmp_path))

    results = load_results(str(tmp_path), environment="mujoco", n_agents=5)
    assert set(results) == {"gated_attn"}
    assert results["gated_attn"][0]["n_agents"] == 5


def test_exporter_prefers_namespaced_checkpoint(tmp_path):
    namespaced = tmp_path / "mpe" / "3_agents" / "commnet_seed0.pt"
    namespaced.parent.mkdir(parents=True)
    namespaced.touch()
    legacy = tmp_path / "commnet_seed0.pt"
    legacy.touch()

    resolved = _checkpoint_path(str(tmp_path), "mpe", 3, "commnet", 0)
    assert resolved == namespaced
