"""Lightweight three-dimensional multi-drone coverage environment.

This environment models each drone as a damped holonomic point mass. It is a
coordination benchmark, not a high-fidelity flight-dynamics simulator. MuJoCo
provides integration, contacts, joint limits, and rendering. The policy issues
axis-aligned velocity commands so learning remains focused on coordination.
"""

from typing import Dict, Optional, Tuple

import mujoco
import numpy as np
import torch

from src.envs.base import validate_actions, validate_n_agents


class MujocoDroneEnv:
    """Cooperative 3D waypoint coverage with range-limited communication."""

    metadata = {"render_modes": (None, "human", "rgb_array")}
    action_names = (
        "hover",
        "+x",
        "-x",
        "+y",
        "-y",
        "+z",
        "-z",
    )

    def __init__(
        self,
        n_agents: int = 3,
        max_steps: int = 150,
        arena_size: float = 4.0,
        flight_height: float = 4.0,
        max_speed: float = 1.5,
        comm_range: float = 3.0,
        max_observed_agents: int = 3,
        max_observed_targets: int = 3,
        collision_penalty: float = 0.25,
        control_penalty: float = 0.002,
        render_mode: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        validate_n_agents(n_agents)
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if arena_size <= 0 or flight_height <= 0:
            raise ValueError("arena_size and flight_height must be positive")
        if max_speed <= 0 or comm_range <= 0:
            raise ValueError("max_speed and comm_range must be positive")
        if max_observed_agents < 0 or max_observed_targets < 1:
            raise ValueError(
                "max_observed_agents must be non-negative and "
                "max_observed_targets must be positive"
            )
        if collision_penalty < 0 or control_penalty < 0:
            raise ValueError("penalties must be non-negative")
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unsupported render_mode: {render_mode!r}")

        self.n_agents = n_agents
        self.n_targets = n_agents
        self.max_steps = max_steps
        self.arena_size = float(arena_size)
        self.flight_height = float(flight_height)
        self.max_speed = float(max_speed)
        self.comm_range = float(comm_range)
        self.max_observed_agents = max_observed_agents
        self.max_observed_targets = max_observed_targets
        self.collision_penalty = float(collision_penalty)
        self.control_penalty = float(control_penalty)
        self.render_mode = render_mode

        self.act_dim = len(self.action_names)
        self.obs_dim = 6 + 3 * max_observed_agents + 4 * max_observed_targets
        self.agent_radius = 0.18
        self.target_half_extent = 0.32
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
        colors = (
            "0.20 0.60 1.00 1",
            "1.00 0.35 0.25 1",
            "0.25 0.85 0.45 1",
            "0.85 0.45 1.00 1",
            "1.00 0.75 0.20 1",
            "0.20 0.90 0.90 1",
        )
        agents = []
        targets = []
        actuators = []

        for index in range(self.n_targets):
            dotted_outline = self._dotted_box_sites(index)
            targets.append(
                f"""
                <body name="target_{index}" pos="0 0 1">
                  <geom name="target_geom_{index}" type="box"
                        size="{self.target_half_extent} {self.target_half_extent} {self.target_half_extent}"
                        rgba="1 0.78 0.12 0.10"
                        contype="0" conaffinity="0"/>
                  {dotted_outline}
                </body>"""
            )

        for index in range(self.n_agents):
            color = colors[index % len(colors)]
            agents.append(
                f"""
                <body name="agent_{index}" pos="0 0 0">
                  <joint name="agent_{index}_x" type="slide" axis="1 0 0"
                         limited="true" range="{-self.arena_size} {self.arena_size}"/>
                  <joint name="agent_{index}_y" type="slide" axis="0 1 0"
                         limited="true" range="{-self.arena_size} {self.arena_size}"/>
                  <joint name="agent_{index}_z" type="slide" axis="0 0 1"
                         limited="true" range="0.35 {self.flight_height}"/>
                  <geom name="agent_body_{index}" type="sphere"
                        size="{self.agent_radius}" mass="0.7" rgba="{color}"/>
                  <geom type="capsule" fromto="-0.32 0 0 0.32 0 0"
                        size="0.035" mass="0.05" rgba="0.12 0.14 0.18 1"/>
                  <geom type="capsule" fromto="0 -0.32 0 0 0.32 0"
                        size="0.035" mass="0.05" rgba="0.12 0.14 0.18 1"/>
                  <site pos="0.32 0 0" size="0.07" rgba="{color}"/>
                  <site pos="-0.32 0 0" size="0.07" rgba="{color}"/>
                  <site pos="0 0.32 0" size="0.07" rgba="{color}"/>
                  <site pos="0 -0.32 0" size="0.07" rgba="{color}"/>
                </body>"""
            )
            for axis in ("x", "y", "z"):
                actuators.append(
                    f'<velocity name="agent_{index}_velocity_{axis}" '
                    f'joint="agent_{index}_{axis}" kv="3.5" '
                    f'ctrlrange="{-self.max_speed} {self.max_speed}" '
                    'ctrllimited="true"/>'
                )

        extent = self.arena_size + 0.5
        camera_z = max(7.0, self.flight_height + 4.5)
        return f"""
        <mujoco model="multi_drone_coverage">
          <compiler angle="radian"/>
          <option timestep="0.04" gravity="0 0 0" integrator="RK4"/>
          <default>
            <joint damping="1.4" armature="0.03"/>
            <geom friction="0.8 0.1 0.1" condim="3"/>
            <site type="sphere"/>
          </default>
          <visual>
            <global offwidth="800" offheight="640"/>
            <rgba haze="0.10 0.13 0.18 1"/>
          </visual>
          <worldbody>
            <light name="key" pos="-3 -4 10" dir="0.3 0.3 -1" diffuse="1 1 1"/>
            <light name="fill" pos="4 2 6" dir="-0.4 -0.2 -1" diffuse="0.4 0.5 0.7"/>
            <camera name="overview" pos="8 -10 {camera_z}"
                    xyaxes="0.78 0.62 0 -0.32 0.40 0.86"/>
            <geom name="floor" type="plane" size="{extent} {extent} 0.1"
                  rgba="0.10 0.13 0.17 1"/>
            <geom name="origin_x" type="capsule" fromto="0 0 0.01 1 0 0.01"
                  size="0.018" rgba="0.9 0.25 0.2 1" contype="0" conaffinity="0"/>
            <geom name="origin_y" type="capsule" fromto="0 0 0.01 0 1 0.01"
                  size="0.018" rgba="0.2 0.8 0.3 1" contype="0" conaffinity="0"/>
            {"".join(targets)}
            {"".join(agents)}
          </worldbody>
          <actuator>{"".join(actuators)}</actuator>
        </mujoco>
        """

    def _dotted_box_sites(self, target_index: int) -> str:
        """Build small marker sites along all twelve edges of a goal region."""
        extent = self.target_half_extent
        coordinates = np.linspace(-extent, extent, num=6)
        points = set()
        for varying_axis in range(3):
            fixed_axes = [axis for axis in range(3) if axis != varying_axis]
            for first_sign in (-extent, extent):
                for second_sign in (-extent, extent):
                    for coordinate in coordinates:
                        point = [0.0, 0.0, 0.0]
                        point[varying_axis] = float(coordinate)
                        point[fixed_axes[0]] = first_sign
                        point[fixed_axes[1]] = second_sign
                        points.add(tuple(point))
        return "".join(
            f'<site name="target_dot_{target_index}_{dot_index}" '
            f'pos="{x:.4f} {y:.4f} {z:.4f}" size="0.035" '
            'rgba="1 0.82 0.18 0.95"/>'
            for dot_index, (x, y, z) in enumerate(sorted(points))
        )

    def _joint_addresses(self, address_kind: str) -> np.ndarray:
        indices = np.empty((self.n_agents, 3), dtype=np.int32)
        addresses = (
            self.model.jnt_qposadr if address_kind == "qpos" else self.model.jnt_dofadr
        )
        for agent_index in range(self.n_agents):
            for axis_index, axis in enumerate(("x", "y", "z")):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"agent_{agent_index}_{axis}",
                )
                indices[agent_index, axis_index] = addresses[joint_id]
        return indices

    def _sample_positions(self, count: int, min_distance: float) -> np.ndarray:
        horizontal_margin = self.agent_radius + 0.2
        low = np.asarray(
            (-self.arena_size + horizontal_margin,) * 2 + (0.6,),
            dtype=np.float64,
        )
        high = np.asarray(
            (self.arena_size - horizontal_margin,) * 2 + (self.flight_height - 0.25,),
            dtype=np.float64,
        )
        if high[2] <= low[2]:
            raise ValueError("flight_height must leave room above the minimum altitude")

        positions = []
        for _ in range(count):
            for _attempt in range(10_000):
                candidate = self._rng.uniform(low, high)
                if all(
                    np.linalg.norm(candidate - existing) >= min_distance
                    for existing in positions
                ):
                    positions.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "could not place entities without overlap; increase arena size"
                )
        return np.asarray(positions, dtype=np.float64)

    @property
    def agent_positions(self) -> np.ndarray:
        """Return 3D agent positions with shape ``(N, 3)``."""
        return self.data.qpos[self._agent_qpos_indices].copy()

    @property
    def agent_velocities(self) -> np.ndarray:
        """Return 3D agent velocities with shape ``(N, 3)``."""
        return self.data.qvel[self._agent_qvel_indices].copy()

    @property
    def target_positions(self) -> np.ndarray:
        """Return 3D target positions with shape ``(N, 3)``."""
        return self.model.body_pos[self._target_body_ids, :3].copy()

    def communication_adjacency(self) -> np.ndarray:
        """Return the symmetric distance-limited physical graph."""
        positions = self.agent_positions
        distances = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        adjacency = distances <= self.comm_range
        np.fill_diagonal(adjacency, False)
        return adjacency

    @staticmethod
    def _graph_is_connected(adjacency: np.ndarray) -> bool:
        visited = {0}
        frontier = [0]
        while frontier:
            current = frontier.pop()
            for neighbor in np.flatnonzero(adjacency[current]):
                neighbor = int(neighbor)
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        return len(visited) == adjacency.shape[0]

    def _add_connectivity_geoms(self, scene) -> int:
        """Append active physical links to a MuJoCo visualization scene."""
        adjacency = self.communication_adjacency()
        positions = self.agent_positions
        connected = self._graph_is_connected(adjacency)
        color = (
            np.asarray((0.20, 0.90, 1.00, 0.90), dtype=np.float32)
            if connected
            else np.asarray((1.00, 0.55, 0.12, 0.90), dtype=np.float32)
        )
        links_added = 0
        for first in range(self.n_agents):
            for second in range(first + 1, self.n_agents):
                if not adjacency[first, second] or scene.ngeom >= scene.maxgeom:
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.zeros(3),
                    np.zeros(3),
                    np.eye(3).reshape(-1),
                    color,
                )
                mujoco.mjv_connector(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    0.018,
                    positions[first],
                    positions[second],
                )
                geom.emission = 0.35
                scene.ngeom += 1
                links_added += 1
        return links_added

    def _enforce_workspace_bounds(self) -> None:
        """Clamp small solver overshoots at the translational joint limits."""
        lower = np.asarray((-self.arena_size, -self.arena_size, 0.35))
        upper = np.asarray((self.arena_size, self.arena_size, self.flight_height))
        positions = self.agent_positions
        velocities = self.agent_velocities
        clipped_positions = np.clip(positions, lower, upper)
        outward = ((positions <= lower) & (velocities < 0)) | (
            (positions >= upper) & (velocities > 0)
        )
        velocities[outward] = 0.0
        self.data.qpos[self._agent_qpos_indices] = clipped_positions
        self.data.qvel[self._agent_qvel_indices] = velocities

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        agent_positions = self._sample_positions(
            self.n_agents, min_distance=3 * self.agent_radius
        )
        target_positions = self._sample_positions(
            self.n_targets, min_distance=2.5 * self.target_half_extent
        )
        self.data.qpos[self._agent_qpos_indices] = agent_positions
        self.data.qvel[self._agent_qvel_indices] = 0.0
        self.model.body_pos[self._target_body_ids, :3] = target_positions
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
        target_offsets = np.abs(positions[:, np.newaxis, :] - targets[np.newaxis, :, :])
        target_covered = np.any(
            np.all(target_offsets <= self.target_half_extent, axis=-1), axis=0
        )
        position_scale = np.asarray(
            (self.arena_size, self.arena_size, self.flight_height)
        )

        observations = []
        for agent_index in range(self.n_agents):
            own_position = positions[agent_index] / position_scale
            own_velocity = np.clip(velocities[agent_index] / self.max_speed, -1.0, 1.0)
            other_indices = [
                index for index in range(self.n_agents) if index != agent_index
            ]
            other_indices.sort(
                key=lambda index: np.linalg.norm(
                    positions[index] - positions[agent_index]
                )
            )
            neighbor_features = np.zeros(3 * self.max_observed_agents)
            for slot, other_index in enumerate(
                other_indices[: self.max_observed_agents]
            ):
                start = 3 * slot
                neighbor_features[start : start + 3] = (
                    positions[other_index] - positions[agent_index]
                ) / (2 * position_scale)

            target_indices = np.argsort(target_distances[agent_index])
            target_features = np.zeros(4 * self.max_observed_targets)
            for slot, target_index in enumerate(
                target_indices[: self.max_observed_targets]
            ):
                start = 4 * slot
                target_features[start : start + 3] = (
                    targets[target_index] - positions[agent_index]
                ) / (2 * position_scale)
                target_features[start + 3] = float(target_covered[target_index])

            observations.append(
                np.concatenate(
                    (own_position, own_velocity, neighbor_features, target_features)
                )
            )
        return torch.as_tensor(np.asarray(observations), dtype=torch.float32)

    def _reward(self, velocity_commands: np.ndarray) -> float:
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
        active_controls = int(
            np.count_nonzero(np.linalg.norm(velocity_commands, axis=1))
        )
        adjacency = self.communication_adjacency()
        physical_links = int(adjacency.sum())
        physical_graph_connected = self._graph_is_connected(adjacency)
        possible_links = self.n_agents * (self.n_agents - 1)

        reward = -float(nearest_agent_distance.mean())
        reward -= self.collision_penalty * collisions / self.n_agents
        reward -= self.control_penalty * active_controls / self.n_agents
        target_offsets = np.abs(
            positions[:, np.newaxis, :] - self.target_positions[np.newaxis, :, :]
        )
        covered = np.any(
            np.all(target_offsets <= self.target_half_extent, axis=-1), axis=0
        )
        self.last_info = {
            "mean_target_distance": float(nearest_agent_distance.mean()),
            "coverage": float(np.mean(covered)),
            "collisions": float(collisions),
            "control_rate": float(active_controls / self.n_agents),
            "physical_comm_rate": float(physical_links / possible_links),
            "physical_graph_connected": float(physical_graph_connected),
            "mean_altitude": float(positions[:, 2].mean()),
        }
        return reward

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """Advance using categorical hover or signed-axis velocity commands."""
        validate_actions(actions, self.n_agents, self.act_dim)
        action_array = actions.detach().cpu().numpy().astype(np.int64, copy=False)
        direction_vectors = np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (-1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 0.0, 1.0),
                (0.0, 0.0, -1.0),
            ),
            dtype=np.float64,
        )
        velocity_commands = self.max_speed * direction_vectors[action_array]
        return self._advance(velocity_commands)

    def step_velocity(
        self, velocity_commands: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool]:
        """Advance using continuous xyz velocity commands in world units/second.

        This interface is ready for a future continuous-action policy. The
        existing PPO trainer continues to use :meth:`step` and its categorical
        action head.
        """
        if velocity_commands.shape != (self.n_agents, 3):
            raise ValueError(
                "velocity_commands must have shape "
                f"({self.n_agents}, 3), got {tuple(velocity_commands.shape)}"
            )
        if not torch.is_floating_point(velocity_commands):
            raise TypeError("velocity_commands must be floating point")
        command_array = velocity_commands.detach().cpu().numpy()
        if not np.isfinite(command_array).all():
            raise ValueError("velocity_commands must be finite")
        if np.any(np.abs(command_array) > self.max_speed):
            raise ValueError(
                f"velocity_commands must lie in [-{self.max_speed}, {self.max_speed}]"
            )
        return self._advance(command_array.astype(np.float64, copy=False))

    def _advance(
        self, velocity_commands: np.ndarray
    ) -> Tuple[torch.Tensor, float, bool]:
        self.data.ctrl[:] = velocity_commands.reshape(-1)
        mujoco.mj_step(self.model, self.data)
        self._enforce_workspace_bounds()
        mujoco.mj_forward(self.model, self.data)
        self._step_count += 1
        reward = self._reward(velocity_commands)
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
                self._renderer = mujoco.Renderer(self.model, height=640, width=800)
            self._renderer.update_scene(self.data, camera="overview")
            self._add_connectivity_geoms(self._renderer.scene)
            return self._renderer.render().copy()

        if self._viewer is None:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self.model, self.data)
            self.configure_viewer_camera(self._viewer.cam)
        with self._viewer.lock():
            self._viewer.user_scn.ngeom = 0
            self._add_connectivity_geoms(self._viewer.user_scn)
        self._viewer.sync()
        return None

    def configure_viewer_camera(self, camera) -> None:
        """Frame the complete flight volume in an interactive free camera."""
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = (0.0, 0.0, 0.45 * self.flight_height)
        camera.distance = 2.9 * max(self.arena_size, self.flight_height)
        camera.azimuth = 135.0
        camera.elevation = -30.0

    def close(self) -> None:
        """Release renderer and interactive viewer resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
