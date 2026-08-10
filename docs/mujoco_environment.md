# MuJoCo point-mass coverage environment

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
