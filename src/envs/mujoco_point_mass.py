"""Lightweight MuJoCo multi-agent coverage environment.

Agents are planar point masses driven by discrete force commands inside a 3D
MuJoCo scene. The task retains the coverage objective of MPE ``simple_spread``
while providing physical dynamics, collisions, scalable local observations,
and a communication graph derived from agent distance.
"""

from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
import torch

from src.envs.base import validate_actions, validate_n_agents


class MujocoPointMassEnv:
    """Cooperative landmark coverage with a configurable number of agents."""

    metadata = {"render_modes": (None, "human", "rgb_array")}

    def __init__(
        self,
        n_agents: int = 3,
        max_steps: int = 100,
        arena_size: float = 4.0,
        force_scale: float = 2.5,
        comm_range: float = 2.5,
        max_observed_agents: int = 3,
        max_observed_targets: int = 3,
        collision_penalty: float = 0.25,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        validate_n_agents(n_agents)
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if arena_size <= 0 or force_scale <= 0 or comm_range <= 0:
            raise ValueError("arena_size, force_scale, and comm_range must be positive")
        if max_observed_agents < 0 or max_observed_targets < 1:
            raise ValueError(
                "max_observed_agents must be non-negative and "
                "max_observed_targets must be positive"
            )
        if collision_penalty < 0:
            raise ValueError("collision_penalty must be non-negative")
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")

        self.n_agents = n_agents
        self.n_targets = n_agents
        self.max_steps = max_steps
        self.arena_size = float(arena_size)
        self.force_scale = float(force_scale)
        self.comm_range = float(comm_range)
        self.max_observed_agents = max_observed_agents
        self.max_observed_targets = max_observed_targets
        self.collision_penalty = float(collision_penalty)
        self.render_mode = render_mode

        self.act_dim = 5
        self.obs_dim = 4 + 2 * self.max_observed_agents + 3 * self.max_observed_targets
        self.agent_radius = 0.16
        self.target_radius = 0.24
        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._renderer = None
        self._viewer = None
        self.last_info: Dict[str, float] = {}

        self.model = mujoco.MjModel.from_xml_string(self._build_xml())
        self.data = mujoco.MjData(self.model)
        self._agent_qpos_indices = self._joint_addresses("qpos")
        self._agent_qvel_indices = self._joint_addresses("qvel")
        self._target_body_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"target_{index}"
                )
                for index in range(self.n_targets)
            ],
            dtype=np.int32,
        )

    def _build_xml(self) -> str:
        half_extent = self.arena_size
        wall_height = 0.35
        wall_thickness = 0.08
        colors = (
            "0.20 0.55 0.95 1",
            "0.95 0.35 0.25 1",
            "0.25 0.75 0.40 1",
            "0.80 0.45 0.90 1",
            "0.95 0.70 0.20 1",
            "0.20 0.80 0.80 1",
        )

        target_bodies = []
        agent_bodies = []
        actuators = []
        for index in range(self.n_targets):
            target_bodies.append(
                f"""
                <body name="target_{index}" pos="0 0 0.025">
                  <geom name="target_geom_{index}" type="cylinder"
                        size="{self.target_radius} 0.025"
                        rgba="0.95 0.78 0.18 0.75"
                        contype="0" conaffinity="0"/>
                </body>"""
            )

        for index in range(self.n_agents):
            color = colors[index % len(colors)]
            agent_bodies.append(
                f"""
                <body name="agent_{index}" pos="0 0 {self.agent_radius}">
                  <joint name="agent_{index}_x" type="slide" axis="1 0 0"
                         limited="true" range="{-half_extent} {half_extent}"/>
                  <joint name="agent_{index}_y" type="slide" axis="0 1 0"
                         limited="true" range="{-half_extent} {half_extent}"/>
                  <geom name="agent_geom_{index}" type="sphere"
                        size="{self.agent_radius}" mass="1" rgba="{color}"/>
                </body>"""
            )
            actuators.extend(
                (
                    f'<motor name="agent_{index}_motor_x" joint="agent_{index}_x" '
                    f'gear="{self.force_scale}" ctrlrange="-1 1" ctrllimited="true"/>',
                    f'<motor name="agent_{index}_motor_y" joint="agent_{index}_y" '
                    f'gear="{self.force_scale}" ctrlrange="-1 1" ctrllimited="true"/>',
                )
            )

        return f"""
        <mujoco model="multi_agent_point_mass">
          <compiler angle="radian"/>
          <option timestep="0.04" gravity="0 0 0" integrator="RK4"/>
          <default>
            <joint damping="1.2" armature="0.02"/>
            <geom friction="0.8 0.1 0.1" condim="3"/>
          </default>
          <visual>
            <global offwidth="640" offheight="640"/>
            <rgba haze="0.15 0.18 0.22 1"/>
          </visual>
          <worldbody>
            <light name="key" pos="0 -2 8" dir="0 0 -1" diffuse="1 1 1"/>
            <camera name="overview" pos="0 0 10" quat="1 0 0 0"/>
            <geom name="floor" type="plane" size="{half_extent + 0.5} {half_extent + 0.5} 0.1"
                  rgba="0.12 0.15 0.18 1"/>
            <geom name="wall_north" type="box"
                  pos="0 {half_extent + wall_thickness} {wall_height}"
                  size="{half_extent + wall_thickness} {wall_thickness} {wall_height}"
                  rgba="0.30 0.34 0.40 1"/>
            <geom name="wall_south" type="box"
                  pos="0 {-half_extent - wall_thickness} {wall_height}"
                  size="{half_extent + wall_thickness} {wall_thickness} {wall_height}"
                  rgba="0.30 0.34 0.40 1"/>
            <geom name="wall_east" type="box"
                  pos="{half_extent + wall_thickness} 0 {wall_height}"
                  size="{wall_thickness} {half_extent + wall_thickness} {wall_height}"
                  rgba="0.30 0.34 0.40 1"/>
            <geom name="wall_west" type="box"
                  pos="{-half_extent - wall_thickness} 0 {wall_height}"
                  size="{wall_thickness} {half_extent + wall_thickness} {wall_height}"
                  rgba="0.30 0.34 0.40 1"/>
            {"".join(target_bodies)}
            {"".join(agent_bodies)}
          </worldbody>
          <actuator>
            {"".join(actuators)}
          </actuator>
        </mujoco>
        """

    def _joint_addresses(self, address_kind: str) -> np.ndarray:
        indices = np.empty((self.n_agents, 2), dtype=np.int32)
        address_array = (
            self.model.jnt_qposadr if address_kind == "qpos" else self.model.jnt_dofadr
        )
        for agent_index in range(self.n_agents):
            for axis_index, axis in enumerate(("x", "y")):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"agent_{agent_index}_{axis}",
                )
                indices[agent_index, axis_index] = address_array[joint_id]
        return indices

    def _sample_positions(self, count: int, min_distance: float) -> np.ndarray:
        margin = max(self.agent_radius, self.target_radius) + 0.1
        low, high = -self.arena_size + margin, self.arena_size - margin
        positions = []
        for _ in range(count):
            for _attempt in range(10_000):
                candidate = self._rng.uniform(low, high, size=2)
                if all(
                    np.linalg.norm(candidate - existing) >= min_distance
                    for existing in positions
                ):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "could not place entities without overlap; increase arena_size"
                )
        return np.asarray(positions, dtype=np.float64)

    @property
    def agent_positions(self) -> np.ndarray:
        """Return a copy of planar agent positions with shape ``(N, 2)``."""
        return self.data.qpos[self._agent_qpos_indices].copy()

    @property
    def agent_velocities(self) -> np.ndarray:
        """Return a copy of planar agent velocities with shape ``(N, 2)``."""
        return self.data.qvel[self._agent_qvel_indices].copy()

    @property
    def target_positions(self) -> np.ndarray:
        """Return a copy of planar landmark positions with shape ``(N, 2)``."""
        return self.model.body_pos[self._target_body_ids, :2].copy()

    def communication_adjacency(self) -> np.ndarray:
        """Return the physical, distance-limited communication graph."""
        positions = self.agent_positions
        distances = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        adjacency = distances <= self.comm_range
        np.fill_diagonal(adjacency, False)
        return adjacency

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        agent_positions = self._sample_positions(
            self.n_agents, min_distance=2.5 * self.agent_radius
        )
        target_positions = self._sample_positions(
            self.n_targets, min_distance=2.5 * self.target_radius
        )
        self.data.qpos[self._agent_qpos_indices] = agent_positions
        self.data.qvel[self._agent_qvel_indices] = 0.0
        self.model.body_pos[self._target_body_ids, :2] = target_positions
        self._step_count = 0
        self.last_info = {}
        mujoco.mj_forward(self.model, self.data)
        return self._observations()

    def _observations(self) -> torch.Tensor:
        positions = self.agent_positions
        velocities = self.agent_velocities
        targets = self.target_positions
        target_distances = np.linalg.norm(
            positions[:, np.newaxis, :] - targets[np.newaxis, :, :], axis=-1
        )
        target_covered = target_distances.min(axis=0) <= self.target_radius

        observations = []
        for agent_index in range(self.n_agents):
            own_position = positions[agent_index] / self.arena_size
            own_velocity = np.clip(velocities[agent_index] / self.force_scale, -1, 1)

            other_indices = [
                index for index in range(self.n_agents) if index != agent_index
            ]
            other_indices.sort(
                key=lambda index: np.linalg.norm(
                    positions[index] - positions[agent_index]
                )
            )
            neighbor_features = np.zeros(2 * self.max_observed_agents)
            for slot, other_index in enumerate(
                other_indices[: self.max_observed_agents]
            ):
                relative = (positions[other_index] - positions[agent_index]) / (
                    2 * self.arena_size
                )
                neighbor_features[2 * slot : 2 * slot + 2] = relative

            target_indices = np.argsort(target_distances[agent_index])
            target_features = np.zeros(3 * self.max_observed_targets)
            for slot, target_index in enumerate(
                target_indices[: self.max_observed_targets]
            ):
                relative = (targets[target_index] - positions[agent_index]) / (
                    2 * self.arena_size
                )
                offset = 3 * slot
                target_features[offset : offset + 2] = relative
                target_features[offset + 2] = float(target_covered[target_index])

            observations.append(
                np.concatenate(
                    (own_position, own_velocity, neighbor_features, target_features)
                )
            )

        return torch.as_tensor(np.asarray(observations), dtype=torch.float32)

    def _reward(self, actions: np.ndarray) -> float:
        positions = self.agent_positions
        target_distances = np.linalg.norm(
            positions[:, np.newaxis, :] - self.target_positions[np.newaxis, :, :],
            axis=-1,
        )
        nearest_agent_distance = target_distances.min(axis=0)

        pairwise_distances = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        collision_matrix = pairwise_distances < 2 * self.agent_radius
        collisions = int(np.triu(collision_matrix, k=1).sum())
        coverage = float(np.mean(nearest_agent_distance <= self.target_radius))

        reward = -float(nearest_agent_distance.mean())
        reward -= self.collision_penalty * collisions / self.n_agents
        physical_links = self.communication_adjacency().sum()
        possible_links = self.n_agents * (self.n_agents - 1)
        self.last_info = {
            "mean_target_distance": float(nearest_agent_distance.mean()),
            "coverage": coverage,
            "collisions": float(collisions),
            "physical_comm_rate": float(physical_links / possible_links),
        }
        return reward

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        validate_actions(actions, self.n_agents, self.act_dim)
        action_array = actions.detach().cpu().numpy().astype(np.int64, copy=False)
        controls = np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)),
            dtype=np.float64,
        )
        self.data.ctrl[:] = controls[action_array].reshape(-1)
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        reward = self._reward(action_array)
        done = self._step_count >= self.max_steps
        observations = self._observations()

        if self.render_mode == "human":
            self.render()
        return observations, reward, done

    def render(self):
        """Render the current scene according to ``render_mode``."""
        if self.render_mode is None:
            raise RuntimeError(
                "set render_mode to 'human' or 'rgb_array' at construction"
            )
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=640, width=640)
            self._renderer.update_scene(self.data, camera="overview")
            return self._renderer.render().copy()

        if self._viewer is None:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
