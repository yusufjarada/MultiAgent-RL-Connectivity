"""Run a visual MuJoCo environment demo with a transparent heuristic policy.

On macOS, MuJoCo's interactive viewer must be launched with ``mjpython``:

    venv/bin/mjpython scripts/demo_mujoco.py --env drone --agents 5
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.envs.mujoco_drone import MujocoDroneEnv
from src.envs.mujoco_point_mass import MujocoPointMassEnv


def target_seeking_actions(env) -> torch.Tensor:
    """Move each agent toward the same-index target along its largest error axis."""
    offsets = env.target_positions - env.agent_positions
    actions = np.zeros(env.n_agents, dtype=np.int64)
    for agent_index, offset in enumerate(offsets):
        axis = int(np.argmax(np.abs(offset)))
        if abs(offset[axis]) < 0.08:
            continue
        direction_offset = 1 if offset[axis] > 0 else 2
        actions[agent_index] = 2 * axis + direction_offset
    return torch.as_tensor(actions, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="View a MuJoCo multi-agent task.")
    parser.add_argument("--env", choices=("planar", "drone"), default="drone")
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument(
        "--realtime-delay",
        type=float,
        default=0.025,
        help="wall-clock delay between simulation steps",
    )
    args = parser.parse_args()

    environment_class = MujocoDroneEnv if args.env == "drone" else MujocoPointMassEnv
    env = environment_class(
        n_agents=args.agents,
        max_steps=args.steps,
        render_mode="human",
        seed=args.seed,
    )
    try:
        env.reset(seed=args.seed)
        env.render()
        for step_index in range(args.steps):
            if env._viewer is not None and not env._viewer.is_running():
                break
            _, _, done = env.step(target_seeking_actions(env))
            if step_index % 25 == 0 and env.last_info:
                info = env.last_info
                status = (
                    "connected"
                    if info.get("physical_graph_connected", 0.0)
                    else "fragmented"
                )
                altitude = info.get("mean_altitude")
                altitude_text = (
                    f" | altitude={altitude:.2f}" if altitude is not None else ""
                )
                print(
                    f"step={step_index:04d} | coverage={info['coverage']:.0%} "
                    f"| links={info['physical_comm_rate']:.0%} | {status} "
                    f"| target_distance={info['mean_target_distance']:.2f}"
                    f"{altitude_text}"
                )
            if done:
                break
            time.sleep(max(0.0, args.realtime_delay))
    finally:
        env.close()


if __name__ == "__main__":
    main()
