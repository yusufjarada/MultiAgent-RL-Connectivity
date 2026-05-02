"""
Train communication methods on MPE simple_spread using PPO.

Usage:
    python scripts/train.py --method commnet --timesteps 200000
    python scripts/train.py --method all --timesteps 200000
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import torch
import numpy as np

from src.envs.mpe_wrapper import MPEWrapper
from src.comm.commnet import CommNet
from src.comm.ic3net import IC3Net
from src.comm.tarmac import TarMAC
from src.comm.gated_attn import GatedAttnComm
from src.training.ppo_trainer import PPOTrainer


def build_comm_module(method: str, obs_dim: int, act_dim: int, n_agents: int,
                      hidden_dim: int = 64, msg_dim: int = 32):
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


def train_method(method: str, n_agents: int, total_timesteps: int, seed: int,
                 results_dir: str):
    print(f"\n{'='*60}")
    print(f"Training: {method} | Agents: {n_agents} | Steps: {total_timesteps} | Seed: {seed}")
    print(f"{'='*60}\n")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MPEWrapper(n_agents=n_agents, max_cycles=25)
    comm = build_comm_module(method, env.obs_dim, env.act_dim, env.n_agents)
    param_count = sum(p.numel() for p in comm.parameters())
    print(f"Model parameters: {param_count:,}")

    trainer = PPOTrainer(
        comm, env,
        lr_actor=3e-4, lr_critic=1e-3,
        gamma=0.99, gae_lambda=0.95,
        clip_eps=0.2, entropy_coef=0.01,
        ppo_epochs=4, batch_size=64,
    )

    start = time.time()
    stats, ep_rewards = trainer.train(
        total_timesteps=total_timesteps,
        rollout_steps=256,
        log_interval=10,
    )
    elapsed = time.time() - start

    print(f"\nTraining complete in {elapsed:.1f}s ({len(ep_rewards)} episodes)")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    result_file = os.path.join(results_dir, f"{method}_seed{seed}.json")

    results = {
        'method': method,
        'n_agents': n_agents,
        'total_timesteps': total_timesteps,
        'seed': seed,
        'param_count': param_count,
        'training_time_s': elapsed,
        'rewards': ep_rewards,
        'comm_rates': [s['avg_comm_rate'] for s in stats],
        'conn_losses': [s['conn_loss'] for s in stats],
        'policy_losses': [s['policy_loss'] for s in stats],
        'value_losses': [s['value_loss'] for s in stats],
        'entropies': [s['entropy'] for s in stats],
    }

    with open(result_file, 'w') as f:
        json.dump(results, f)
    print(f"Results saved to {result_file}")

    model_file = os.path.join(results_dir, f"{method}_seed{seed}.pt")
    torch.save(comm.state_dict(), model_file)

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all',
                        choices=['commnet', 'ic3net', 'tarmac', 'gated_attn', 'all'])
    parser.add_argument('--agents', type=int, default=3)
    parser.add_argument('--timesteps', type=int, default=200000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()

    methods = ['commnet', 'ic3net', 'tarmac', 'gated_attn'] if args.method == 'all' else [args.method]

    for method in methods:
        for seed in args.seeds:
            train_method(method, args.agents, args.timesteps, seed, args.results_dir)

    print(f"\n{'='*60}")
    print("All training complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
