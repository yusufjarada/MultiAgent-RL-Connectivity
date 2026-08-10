"""
Run trained policies on MPE and export trajectories for the browser demo.

Saves agent positions, gate states, Fiedler values, and rewards
per timestep as JSON that main.js can replay.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.comm.factory import build_comm_module
from src.envs.mpe_wrapper import MPEWrapper
from src.training.checkpoint import load_actor_state
from src.utils.graph import fiedler_value


def record_episode(comm, env, method):
    """Run one episode and record everything."""
    obs = env.reset()
    frames = []
    done = False

    # Get the raw env to extract positions
    raw_env = env.env

    while not done:
        obs_batch = obs.unsqueeze(0)

        with torch.no_grad():
            logits, info = comm(obs_batch, hard_gate=True)

        logits = logits.squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()

        # Extract agent positions from the raw environment
        agent_positions = []
        for agent in raw_env.unwrapped.world.agents:
            agent_positions.append(agent.state.p_pos.tolist())

        landmark_positions = []
        for landmark in raw_env.unwrapped.world.landmarks:
            landmark_positions.append(landmark.state.p_pos.tolist())

        # Build edge list and compute Fiedler
        n = env.n_agents
        edges = []
        if method == "commnet" or method == "tarmac":
            # Broadcast: all pairs connected
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append([i, j])
        elif method == "ic3net":
            gates = info["gates"].squeeze(0).cpu().numpy()  # (n,)
            for i in range(n):
                for j in range(i + 1, n):
                    if gates[i] > 0.5 or gates[j] > 0.5:
                        edges.append([i, j])
        elif method == "gated_attn":
            gates = info["gates"].squeeze(0).cpu().numpy()  # (n, n)
            for i in range(n):
                for j in range(i + 1, n):
                    if gates[i][j] > 0.5 or gates[j][i] > 0.5:
                        edges.append([i, j])

        # Fiedler value
        adj = np.zeros((n, n))
        for i, j in edges:
            adj[i][j] = adj[j][i] = 1
        fv = float(fiedler_value(adj)) if n >= 2 else 0.0

        comm_rate = len(edges) / max(n * (n - 1) / 2, 1)

        frame = {
            "agents": agent_positions,
            "landmarks": landmark_positions,
            "edges": edges,
            "fiedler": round(fv, 4),
            "comm_rate": round(comm_rate, 4),
        }
        frames.append(frame)

        obs_next, reward, done = env.step(actions)
        obs = obs_next

    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="demo/trajectories.json")
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = MPEWrapper(n_agents=args.agents, max_cycles=25, seed=args.seed)
    methods = ["commnet", "ic3net", "tarmac", "gated_attn"]

    all_trajectories = {}

    for method in methods:
        model_file = (
            Path(args.results_dir)
            / "mpe"
            / f"{args.agents}_agents"
            / f"{method}_seed{args.seed}.pt"
        )
        legacy_model_file = Path(args.results_dir) / f"{method}_seed{args.seed}.pt"
        if not model_file.exists() and legacy_model_file.exists():
            model_file = legacy_model_file
        if not model_file.exists():
            print(f"Skipping {method} — no model found at {model_file}")
            continue

        comm = build_comm_module(method, env.obs_dim, env.act_dim, env.n_agents)
        comm.load_state_dict(load_actor_state(model_file))
        comm.eval()

        best_frames = None

        for ep in range(args.episodes):
            torch.manual_seed(ep + 42)
            np.random.seed(ep + 42)
            frames = record_episode(comm, env, method)

            # Rough reward proxy: we want diverse, interesting trajectories
            # Pick the longest episode (more timesteps = more interesting)
            if len(frames) > (len(best_frames) if best_frames else 0):
                best_frames = frames

        all_trajectories[method] = best_frames
        print(f"{method}: recorded {len(best_frames)} frames")

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output:
        json.dump(all_trajectories, output)

    print(f"\nTrajectories saved to {output_file}")
    env.close()


if __name__ == "__main__":
    main()
