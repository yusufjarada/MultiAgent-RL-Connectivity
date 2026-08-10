"""Factories for supported multi-agent environments."""

from typing import Optional

from src.envs.base import MultiAgentEnv

ENVIRONMENTS = ("mpe", "mujoco", "mujoco_drone")


def make_env(
    name: str,
    n_agents: int,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    render_mode: Optional[str] = None,
) -> MultiAgentEnv:
    """Construct an environment without importing unused simulator stacks."""
    if name == "mpe":
        if render_mode is not None:
            raise ValueError("the MPE tensor wrapper does not expose rendering")
        from src.envs.mpe_wrapper import MPEWrapper

        return MPEWrapper(
            n_agents=n_agents,
            max_cycles=max_steps if max_steps is not None else 25,
            seed=seed,
        )
    if name == "mujoco":
        from src.envs.mujoco_point_mass import MujocoPointMassEnv

        return MujocoPointMassEnv(
            n_agents=n_agents,
            max_steps=max_steps if max_steps is not None else 100,
            seed=seed,
            render_mode=render_mode,
        )
    if name == "mujoco_drone":
        from src.envs.mujoco_drone import MujocoDroneEnv

        return MujocoDroneEnv(
            n_agents=n_agents,
            max_steps=max_steps if max_steps is not None else 150,
            seed=seed,
            render_mode=render_mode,
        )

    choices = ", ".join(ENVIRONMENTS)
    raise ValueError(f"unknown environment {name!r}; choose {choices}")
