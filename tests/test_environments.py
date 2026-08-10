"""Contract tests for MPE and MuJoCo environments."""

import mujoco
import numpy as np
import pytest
import torch

from src.envs.factory import make_env
from src.envs.mujoco_drone import MujocoDroneEnv
from src.envs.mujoco_point_mass import MujocoPointMassEnv


@pytest.mark.parametrize("environment_name", ("mpe", "mujoco", "mujoco_drone"))
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


def test_drone_environment_is_three_dimensional_and_reproducible():
    env = MujocoDroneEnv(n_agents=4)
    try:
        first_observations = env.reset(seed=27)
        first_agents = env.agent_positions
        first_targets = env.target_positions
        second_observations = env.reset(seed=27)

        assert first_agents.shape == (4, 3)
        assert first_targets.shape == (4, 3)
        assert torch.equal(first_observations, second_observations)
        assert np.array_equal(first_agents, env.agent_positions)
        assert np.array_equal(first_targets, env.target_positions)
    finally:
        env.close()


def test_drone_velocity_commands_move_on_each_axis():
    env = MujocoDroneEnv(n_agents=3, comm_range=100.0)
    try:
        env.reset(seed=9)
        start = env.agent_positions.copy()
        actions = torch.tensor((1, 3, 5), dtype=torch.long)
        for _ in range(8):
            env.step(actions)

        displacement = env.agent_positions - start
        assert displacement[0, 0] > 0
        assert displacement[1, 1] > 0
        assert displacement[2, 2] > 0
        assert env.communication_adjacency().sum() == 6
        assert set(env.last_info) == {
            "mean_target_distance",
            "coverage",
            "collisions",
            "control_rate",
            "physical_comm_rate",
            "physical_graph_connected",
            "mean_altitude",
        }
    finally:
        env.close()


def test_drone_connectivity_lines_match_physical_adjacency():
    env = MujocoDroneEnv(n_agents=3, comm_range=3.1)
    try:
        env.reset(seed=4)
        env.data.qpos[env._agent_qpos_indices] = np.asarray(
            ((-3.0, 0.0, 1.0), (0.0, 0.0, 1.0), (3.0, 0.0, 1.0))
        )
        env.data.qvel[env._agent_qvel_indices] = 0.0
        env.step(torch.zeros(3, dtype=torch.long))

        adjacency = env.communication_adjacency()
        assert adjacency[0, 1] and adjacency[1, 2]
        assert not adjacency[0, 2]
        assert env._graph_is_connected(adjacency)
        scene = mujoco.MjvScene(env.model, maxgeom=20)
        assert env._add_connectivity_geoms(scene) == 2
        assert scene.ngeom == 2
        assert np.allclose(scene.geoms[0].rgba[:3], (0.20, 0.90, 1.00))
    finally:
        env.close()


def test_drone_altitude_limits_hold_under_sustained_commands():
    env = MujocoDroneEnv(n_agents=2, flight_height=2.0)
    try:
        env.reset(seed=6)
        for _ in range(250):
            env.step(torch.tensor((5, 6), dtype=torch.long))

        altitudes = env.agent_positions[:, 2]
        assert np.all(altitudes >= 0.35 - 1e-6)
        assert np.all(altitudes <= 2.0 + 1e-6)
    finally:
        env.close()


def test_drone_continuous_velocity_interface_supports_diagonal_flight():
    env = MujocoDroneEnv(n_agents=2, max_speed=1.5)
    try:
        env.reset(seed=18)
        start = env.agent_positions.copy()
        commands = torch.tensor(((0.8, -0.6, 0.4), (-0.5, 0.7, -0.3)))
        for _ in range(12):
            env.step_velocity(commands)

        displacement = env.agent_positions - start
        assert np.all(np.sign(displacement) == np.sign(commands.numpy()))
        with pytest.raises(ValueError, match="shape"):
            env.step_velocity(torch.zeros(2, 2))
        with pytest.raises(ValueError, match="must lie"):
            env.step_velocity(torch.full((2, 3), 2.0))
    finally:
        env.close()


def test_drone_rgb_render_contains_scene():
    env = MujocoDroneEnv(n_agents=3, comm_range=100.0, render_mode="rgb_array")
    try:
        env.reset(seed=5)
        frame = env.render()
        assert frame.shape == (640, 800, 3)
        assert frame.dtype == np.uint8
        assert frame.max() > frame.min()
    finally:
        env.close()


def test_drone_interactive_camera_frames_flight_volume():
    env = MujocoDroneEnv(n_agents=3, arena_size=5.0, flight_height=3.0)
    try:
        camera = mujoco.MjvCamera()
        env.configure_viewer_camera(camera)

        assert camera.type == mujoco.mjtCamera.mjCAMERA_FREE
        assert np.allclose(camera.lookat, (0.0, 0.0, 1.35))
        assert camera.distance == pytest.approx(14.5)
        assert camera.azimuth == pytest.approx(135.0)
        assert camera.elevation == pytest.approx(-30.0)
    finally:
        env.close()
