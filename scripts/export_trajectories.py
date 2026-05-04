"""
Run trained policies on MPE and export trajectories for the browser demo.

Saves agent positions, gate states, Fiedler values, and rewards
per timestep as JSON that main.js can replay.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import inspect
import torch
import numpy as np

from src.envs.mpe_wrapper import MPEWrapper
from src.comm.commnet import CommNet
from src.comm.ic3net import IC3Net
from src.comm.tarmac import TarMAC
from src.comm.gated_attn import GatedAttnComm
from src.utils.graph import fiedler_value, build_adjacency_from_gates


def build_comm_module(method, obs_dim, act_dim, n_agents,
                      hidden_dim=64, msg_dim=32):
    if method == 'commnet':
        return CommNet(obs_dim, hidden_dim, msg_dim, act_dim, n_agents)
    elif method == 'ic3net':
        return IC3Net(obs_dim, hidden_dim, msg_dim, act_dim, n_agents)
    elif method == 'tarmac':
        return TarMAC(obs_dim, hidden_dim, msg_dim, act_dim, n_agents, n_heads=4)
    elif method == 'gated_attn':
        return GatedAttnComm(obs_dim, hidden_dim, msg_dim, act_dim, n_agents,
                             n_heads=4, connectivity_weight=0.5)
    else:
        raise ValueError(f"Unknown method: {method}")


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
            sig = inspect.signature(comm.forward)
            if 'hard_gate' in sig.parameters:
                logits, info = comm(obs_batch, hard_gate=True)
            else:
                logits, info = comm(obs_batch)

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
        if method == 'commnet' or method == 'tarmac':
            # Broadcast: all pairs connected
            for i in range(n):
                for j in range(i + 1, n):
                    edges.append([i, j])
        elif method == 'ic3net':
            gates = info['gates'].squeeze(0).cpu().numpy()  # (n,)
            for i in range(n):
                for j in range(i + 1, n):
                    if gates[i] > 0.5 or gates[j] > 0.5:
                        edges.append([i, j])
        elif method == 'gated_attn':
            gates = info['gates'].squeeze(0).cpu().numpy()  # (n, n)
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
            'agents': agent_positions,
            'landmarks': landmark_positions,
            'edges': edges,
            'fiedler': round(fv, 4),
            'comm_rate': round(comm_rate, 4),
        }
        frames.append(frame)

        obs_next, reward, done = env.step(actions)
        obs = obs_next

    return frames


def main():
    n_agents = 3
    n_episodes = 5  # Record 5 episodes per method, pick the best
    results_dir = 'results'
    output_file = 'demo/trajectories.json'

    env = MPEWrapper(n_agents=n_agents, max_cycles=25)
    methods = ['commnet', 'ic3net', 'tarmac', 'gated_attn']

    all_trajectories = {}

    for method in methods:
        model_file = os.path.join(results_dir, f"{method}_seed0.pt")
        if not os.path.exists(model_file):
            print(f"Skipping {method} — no model found at {model_file}")
            continue

        comm = build_comm_module(method, env.obs_dim, env.act_dim, env.n_agents)
        comm.load_state_dict(torch.load(model_file, weights_only=True))
        comm.eval()

        best_frames = None
        best_reward = float('-inf')

        for ep in range(n_episodes):
            torch.manual_seed(ep + 42)
            np.random.seed(ep + 42)
            frames = record_episode(comm, env, method)

            # Rough reward proxy: we want diverse, interesting trajectories
            # Pick the longest episode (more timesteps = more interesting)
            if len(frames) > (len(best_frames) if best_frames else 0):
                best_frames = frames

        all_trajectories[method] = best_frames
        print(f"{method}: recorded {len(best_frames)} frames")

    with open(output_file, 'w') as f:
        json.dump(all_trajectories, f)

    print(f"\nTrajectories saved to {output_file}")
    env.close()


if __name__ == '__main__':
    main()
