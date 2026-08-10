"""
Export trained PyTorch models to JSON so they can run in the browser demo.

Each model is just a few linear layers — we export the weight matrices
and biases, then reimplement the forward pass in JavaScript.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

from src.comm.factory import COMM_METHODS, build_comm_module
from src.envs.factory import ENVIRONMENTS, make_env
from src.training.checkpoint import load_actor_state


def _checkpoint_path(
    results_dir: str, environment: str, n_agents: int, method: str, seed: int
) -> Path:
    namespaced = (
        Path(results_dir)
        / environment
        / f"{n_agents}_agents"
        / f"{method}_seed{seed}.pt"
    )
    legacy = Path(results_dir) / f"{method}_seed{seed}.pt"
    return namespaced if namespaced.exists() else legacy


def export_model(method, model_file: Path, obs_dim, act_dim, n_agents):
    if not model_file.exists():
        print(f"Skipping {method} — no model at {model_file}")
        return None

    comm = build_comm_module(method, obs_dim, act_dim, n_agents)
    comm.load_state_dict(load_actor_state(model_file))
    comm.eval()

    # Export all parameters as nested dict of lists
    params = {}
    for name, param in comm.named_parameters():
        params[name] = param.detach().cpu().numpy().tolist()

    print(f"  {method}: {len(params)} parameter tensors exported")
    return params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output", default="demo/models.json")
    parser.add_argument(
        "--env", dest="environment", choices=ENVIRONMENTS, default="mpe"
    )
    parser.add_argument("--agents", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(args.environment, n_agents=args.agents, seed=args.seed)
    obs_dim, act_dim = env.obs_dim, env.act_dim
    env.close()

    all_models = {}

    for method in COMM_METHODS:
        model_file = _checkpoint_path(
            args.results_dir, args.environment, args.agents, method, args.seed
        )
        params = export_model(method, model_file, obs_dim, act_dim, args.agents)
        if params:
            all_models[method] = {
                "params": params,
                "method": method,
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "n_agents": args.agents,
                "hidden_dim": 64,
                "msg_dim": 32,
            }

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as output:
        json.dump(all_models, output)

    size_kb = output_file.stat().st_size / 1024
    print(f"\nExported to {output_file} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
