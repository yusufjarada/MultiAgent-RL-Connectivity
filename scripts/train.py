"""
Train communication methods on a supported multi-agent environment using PPO.

Usage:
    python scripts/train.py --env mujoco --agents 5 --method commnet
    python scripts/train.py --env mpe --method all --timesteps 200000
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.comm.factory import COMM_METHODS, build_comm_module
from src.envs.factory import ENVIRONMENTS, make_env
from src.training.checkpoint import save_training_checkpoint
from src.training.ppo_trainer import PPOTrainer


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, default=_json_default)
            output.write("\n")
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _json_default(value):
    """Convert NumPy values returned by simulators to standard JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize value of type {type(value).__name__}")


def train_method(
    method: str,
    n_agents: int,
    total_timesteps: int,
    seed: int,
    results_dir: str,
    environment_name: str = "mpe",
    max_steps: int = None,
    rollout_steps: int = 256,
    ppo_epochs: int = 4,
    batch_size: int = 64,
):
    print(f"\n{'=' * 60}")
    print(
        f"Training: {method} | Environment: {environment_name} | "
        f"Agents: {n_agents} | Steps: {total_timesteps} | Seed: {seed}"
    )
    print(f"{'=' * 60}\n")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_env(environment_name, n_agents=n_agents, max_steps=max_steps, seed=seed)
    comm = build_comm_module(method, env.obs_dim, env.act_dim, env.n_agents)
    param_count = sum(p.numel() for p in comm.parameters())
    print(f"Model parameters: {param_count:,}")

    trainer = PPOTrainer(
        comm,
        env,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        ppo_epochs=ppo_epochs,
        batch_size=batch_size,
    )

    start = time.time()
    stats, ep_rewards = trainer.train(
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        log_interval=10,
    )
    elapsed = time.time() - start

    print(f"\nTraining complete in {elapsed:.1f}s ({len(ep_rewards)} episodes)")

    # Save results
    run_dir = Path(results_dir) / environment_name / f"{n_agents}_agents"
    result_file = run_dir / f"{method}_seed{seed}.json"

    results = {
        "method": method,
        "environment": environment_name,
        "n_agents": n_agents,
        "obs_dim": env.obs_dim,
        "act_dim": env.act_dim,
        "max_steps": env.max_steps,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "param_count": param_count,
        "training_time_s": elapsed,
        "rewards": ep_rewards,
        "comm_rates": [s["avg_comm_rate"] for s in stats],
        "conn_losses": [s["conn_loss"] for s in stats],
        "policy_losses": [s["policy_loss"] for s in stats],
        "value_losses": [s["value_loss"] for s in stats],
        "entropies": [s["entropy"] for s in stats],
    }

    _write_json_atomic(result_file, results)
    print(f"Results saved to {result_file}")

    model_file = run_dir / f"{method}_seed{seed}.pt"
    checkpoint_metadata = {
        key: results[key]
        for key in (
            "method",
            "environment",
            "n_agents",
            "obs_dim",
            "act_dim",
            "max_steps",
            "total_timesteps",
            "seed",
        )
    }
    save_training_checkpoint(
        model_file,
        actor=comm,
        critic=trainer.critic,
        actor_optimizer=trainer.actor_optim,
        critic_optimizer=trainer.critic_optim,
        metadata=checkpoint_metadata,
    )
    print(f"Checkpoint saved to {model_file}")

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train sparse communication policies with multi-agent PPO."
    )
    parser.add_argument(
        "--method", type=str, default="all", choices=[*COMM_METHODS, "all"]
    )
    parser.add_argument(
        "--env", dest="environment_name", choices=ENVIRONMENTS, default="mpe"
    )
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=256)
    args = parser.parse_args()

    methods = list(COMM_METHODS) if args.method == "all" else [args.method]

    for method in methods:
        for seed in args.seeds:
            train_method(
                method,
                args.agents,
                args.timesteps,
                seed,
                args.results_dir,
                environment_name=args.environment_name,
                max_steps=args.max_steps,
                rollout_steps=args.rollout_steps,
            )

    print(f"\n{'=' * 60}")
    print("All training complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
