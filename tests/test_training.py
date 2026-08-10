"""Smoke tests for variable-size PPO training and checkpoint persistence."""

import json

import pytest
import torch

from scripts.train import train_method
from src.comm.factory import build_comm_module
from src.envs.factory import make_env
from src.training.ppo_trainer import PPOTrainer, ValueNetwork


def test_critic_is_permutation_invariant_and_team_size_agnostic():
    critic = ValueNetwork(obs_dim=7, hidden_dim=16)
    observations = torch.randn(3, 5, 7)
    permutation = torch.tensor((2, 0, 4, 1, 3))

    original = critic(observations)
    permuted = critic(observations[:, permutation])
    smaller_team = critic(torch.randn(3, 2, 7))

    assert original.shape == (3,)
    assert smaller_team.shape == (3,)
    assert torch.allclose(original, permuted, atol=1e-6)


@pytest.mark.parametrize("environment_name", ("mpe", "mujoco", "mujoco_drone"))
@pytest.mark.parametrize("n_agents", (2, 5))
def test_ppo_smoke(environment_name, n_agents):
    env = make_env(environment_name, n_agents=n_agents, max_steps=5, seed=2)
    try:
        actor = build_comm_module(
            "gated_attn", env.obs_dim, env.act_dim, n_agents, hidden_dim=16, msg_dim=8
        )
        trainer = PPOTrainer(actor, env, ppo_epochs=1, batch_size=4, entropy_coef=0.0)
        stats, rewards = trainer.train(
            total_timesteps=8, rollout_steps=4, log_interval=100
        )

        assert len(stats) == 2
        assert stats[-1]["timesteps"] == 8
        assert all(
            np_value == np_value
            for np_value in (
                stats[-1]["policy_loss"],
                stats[-1]["value_loss"],
                stats[-1]["conn_loss"],
            )
        )
        assert rewards
    finally:
        env.close()


@pytest.mark.parametrize("environment_name", ("mpe", "mujoco", "mujoco_drone"))
def test_train_method_saves_namespaced_results_and_full_checkpoint(
    tmp_path, environment_name
):
    results = train_method(
        method="commnet",
        n_agents=2,
        total_timesteps=4,
        seed=3,
        results_dir=str(tmp_path),
        environment_name=environment_name,
        max_steps=2,
        rollout_steps=2,
        ppo_epochs=1,
        batch_size=2,
    )
    run_dir = tmp_path / environment_name / "2_agents"
    result_path = run_dir / "commnet_seed3.json"
    checkpoint_path = run_dir / "commnet_seed3.pt"

    assert result_path.exists()
    assert checkpoint_path.exists()
    assert json.loads(result_path.read_text())["environment"] == environment_name
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    assert checkpoint["checkpoint_version"] == 1
    assert checkpoint["metadata"]["n_agents"] == 2
    assert "actor_state_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert results["total_timesteps"] == 4
