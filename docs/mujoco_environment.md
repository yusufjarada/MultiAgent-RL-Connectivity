# MuJoCo coordination environments

The project provides two deliberately lightweight physics-backed tasks:

- `mujoco`: planar point-mass coverage, retained as the stable benchmark;
- `mujoco_drone`: three-dimensional holonomic drone coverage.

Neither environment attempts to reproduce a specific robot's low-level
dynamics. They isolate multi-agent coordination and communication while MuJoCo
provides deterministic integration, contacts, limits, and professional 3D
visualization.

## Quick visual demo

On macOS, launch the interactive viewer through MuJoCo's `mjpython` executable:

```bash
venv/bin/mjpython scripts/demo_mujoco.py --env drone --agents 5
```

Use `--env planar` to view the original task. The demo policy is an intentionally
simple target seeker and is not a trained result.

## 3D drone coverage

Each drone has three translational joints and state `(x, y, z, vx, vy, vz)`.
The categorical policy chooses one of seven target-velocity commands: hover or
motion along the positive or negative x, y, or z axis. This interface works with
the existing PPO implementation without pretending to model quadrotor attitude,
rotors, or aerodynamics.

Every drone is assigned a visible target region represented by a translucent
cube with a dotted outline. A region is covered when a drone lies inside its
axis-aligned bounds. Distance to target centers supplies a smooth learning
signal before coverage.

The standard `step()` method maps the existing categorical policy to the seven
commands. `step_velocity()` accepts a continuous `(N, 3)` tensor of simultaneous
xyz velocity targets, providing a tested interface for a future continuous
policy without forcing that trainer change into the environment work.

The fixed-width local observation contains normalized position and velocity,
relative 3D positions of the nearest teammates, and relative 3D target positions
plus their coverage state. Communication adjacency is computed from full 3D
Euclidean distance. The viewer draws active links directly from that adjacency:
cyan indicates a connected team graph, while amber indicates that the current
physical graph is fragmented. Console telemetry reports the same state together
with coverage, link utilization, altitude, and mean target distance.

Train it with:

```bash
python scripts/train.py --env mujoco_drone --agents 5 --method commnet \
  --timesteps 200000 --seeds 0 1 2
```

## Planar point-mass coverage

`MujocoPointMassEnv` is the first physics-backed task for this project. It is
deliberately lightweight: the research variable is communication, not a
difficult low-level locomotion controller.

## Task

The scene contains `N` planar point-mass agents and `N` stationary landmarks
inside a bounded MuJoCo arena. Agents receive discrete force commands and must
jointly cover the landmarks while avoiding one another.

The shared reward is

\[
r_t = -\frac{1}{N}\sum_{k=1}^{N}\min_i \lVert p_i-p_k^{target}\rVert_2
      - \frac{c}{N} n_{collisions}.
\]

This matches the central coordination pressure in MPE `simple_spread`: sending
every agent to the same landmark performs poorly.

## Actions

Every agent chooses one of five discrete actions:

| Index | Command |
|---:|---|
| 0 | No force |
| 1 | Positive x force |
| 2 | Negative x force |
| 3 | Positive y force |
| 4 | Negative y force |

MuJoCo integrates the resulting motion with joint damping, contact, and arena
boundaries.

## Observations

Observation width does not depend on team size. Each agent receives:

- its normalized planar position and velocity;
- relative positions of up to three nearest teammates;
- relative positions and coverage state of up to three nearest landmarks.

Missing neighbor or landmark slots are zero padded. The policy and the
permutation-invariant centralized critic can therefore process different team
sizes without rebuilding their learnable layers, provided the environment uses
the same observation-slot configuration.

## Physical communication graph

The environment exposes `communication_adjacency()`. For distinct agents,

\[
A^{physical}_{ij} = \mathbb{1}[\lVert p_i-p_j\rVert_2 \le r_{comm}].
\]

Self-links are always disabled. This graph is environment state, separate from
the policy's learned communication gates. A later milestone will combine this
mask with learned gates and add obstacle-based line-of-sight restrictions.

## Reproducibility and rendering

`reset(seed=...)` deterministically regenerates agent and landmark positions.
Use `render_mode="rgb_array"` for recorded evaluation frames or
`render_mode="human"` for an interactive MuJoCo window. Headless training does
not initialize a renderer.
