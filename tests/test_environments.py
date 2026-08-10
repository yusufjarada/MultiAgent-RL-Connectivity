"""Contract tests for MPE and MuJoCo environments."""

import numpy as np
import pytest
import torch

from src.envs.factory import make_env
from src.envs.mujoco_point_mass import MujocoPointMassEnv


@pytest.mark.parametrize("environment_name", ("mpe", "mujoco"))
@pytest.mark.parametrize("n_agents", (2, 5))
def test_environment_contract(environment_name, n_agents):
    env = make_env(environment_name, n_agents=n_agents, max_steps=4, seed=7)
    try:
        observations = env.reset(seed=11)
        assert observations.shape == (n_agents, env.obs_dim)
        assert observations.dtype == torch.float32
        assert torch.isfinite(observations).all()

        for step_index in range(4):
            observations, reward, done = env.step(
                torch.zeros(n_agents, dtype=torch.long)
            )
            assert observations.shape == (n_agents, env.obs_dim)
            assert np.isfinite(reward)
            assert done is (step_index == 3)
    finally:
        env.close()


def test_mujoco_observation_width_is_independent_of_team_size():
    small = MujocoPointMassEnv(n_agents=2)
    large = MujocoPointMassEnv(n_agents=8)
    try:
        assert small.obs_dim == large.obs_dim
        assert small.reset().shape[1] == large.reset().shape[1]
    finally:
        small.close()
        large.close()


def test_mujoco_reset_is_reproducible():
    env = MujocoPointMassEnv(n_agents=5)
    try:
        first = env.reset(seed=123)
        first_agents = env.agent_positions
        first_targets = env.target_positions
        second = env.reset(seed=123)

        assert torch.equal(first, second)
        assert np.array_equal(first_agents, env.agent_positions)
        assert np.array_equal(first_targets, env.target_positions)
    finally:
        env.close()


def test_mujoco_actions_drive_agents_and_graph_is_well_formed():
    env = MujocoPointMassEnv(n_agents=3, comm_range=100.0)
    try:
        env.reset(seed=9)
        starting_x = env.agent_positions[0, 0]
        actions = torch.tensor((1, 0, 0), dtype=torch.long)
        for _ in range(5):
            env.step(actions)

        assert env.agent_positions[0, 0] > starting_x
        adjacency = env.communication_adjacency()
        assert adjacency.shape == (3, 3)
        assert np.array_equal(adjacency, adjacency.T)
        assert not adjacency.diagonal().any()
        assert adjacency.sum() == 6
    finally:
        env.close()


def test_environment_rejects_invalid_actions():
    env = MujocoPointMassEnv(n_agents=3)
    try:
        env.reset()
        with pytest.raises(ValueError, match="shape"):
            env.step(torch.zeros(2, dtype=torch.long))
        with pytest.raises(ValueError, match="must be in"):
            env.step(torch.tensor((0, 1, 5), dtype=torch.long))
        with pytest.raises(TypeError, match="integer"):
            env.step(torch.zeros(3))
    finally:
        env.close()


def test_mujoco_rgb_render():
    env = MujocoPointMassEnv(n_agents=3, render_mode="rgb_array")
    try:
        env.reset(seed=5)
        frame = env.render()
        assert frame.shape == (640, 640, 3)
        assert frame.dtype == np.uint8
        assert frame.max() > frame.min()
    finally:
        env.close()
