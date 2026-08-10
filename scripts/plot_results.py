"""
Plot training results. Generates figures for the paper.

Usage:
    python scripts/plot_results.py --results-dir results
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.envs.factory import ENVIRONMENTS


def load_results(
    results_dir: str, environment: str = None, n_agents: int = None
) -> dict:
    """Load one compatible experiment set, grouped by method.

    Mixing environments or team sizes in the same plot would produce invalid
    comparisons, so callers must filter when the results tree contains more
    than one configuration.
    """
    results = {}
    configurations = set()
    for result_path in Path(results_dir).rglob("*.json"):
        with result_path.open(encoding="utf-8") as input_file:
            data = json.load(input_file)

        run_environment = data.get("environment", "mpe")
        run_n_agents = int(data.get("n_agents", 3))
        if environment is not None and run_environment != environment:
            continue
        if n_agents is not None and run_n_agents != n_agents:
            continue

        configurations.add((run_environment, run_n_agents))
        method = data["method"]
        results.setdefault(method, []).append(data)

    if len(configurations) > 1:
        available = ", ".join(
            f"{env}/{agents}_agents" for env, agents in sorted(configurations)
        )
        raise ValueError(
            "results contain multiple environment/team-size configurations "
            f"({available}); pass --env and --agents to select one"
        )
    return results


def smooth(values, window=50):
    """Running average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


METHOD_COLORS = {
    "commnet": "#2196F3",
    "ic3net": "#FF9800",
    "tarmac": "#9C27B0",
    "gated_attn": "#4CAF50",
}

METHOD_LABELS = {
    "commnet": "CommNet",
    "ic3net": "IC3Net",
    "tarmac": "TarMAC",
    "gated_attn": "Ours (Gated+Conn)",
}


def plot_rewards(results: dict, save_path: str):
    """Plot training reward curves."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, runs in results.items():
        color = METHOD_COLORS.get(method, "#888")
        label = METHOD_LABELS.get(method, method)

        # Average across seeds — smooth per-episode rewards
        all_rewards = [smooth(np.array(r["rewards"]), window=100) for r in runs]
        min_len = min(len(r) for r in all_rewards)
        all_rewards = np.array([r[:min_len] for r in all_rewards])

        mean = all_rewards.mean(axis=0)
        std = all_rewards.std(axis=0)

        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, label=label, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Episode Reward (smoothed)", fontsize=12)
    ax.set_title("Training Reward Comparison", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_comm_rates(results: dict, save_path: str):
    """Plot communication rate over training."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for method, runs in results.items():
        color = METHOD_COLORS.get(method, "#888")
        label = METHOD_LABELS.get(method, method)

        all_rates = [smooth(np.array(r["comm_rates"])) for r in runs]
        min_len = min(len(r) for r in all_rates)
        all_rates = np.array([r[:min_len] for r in all_rates])

        mean = all_rates.mean(axis=0)
        std = all_rates.std(axis=0)

        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, label=label, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Communication Rate", fontsize=12)
    ax.set_title("Communication Rate Over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_pareto(results: dict, save_path: str):
    """Pareto plot: final reward vs final comm rate."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for method, runs in results.items():
        color = METHOD_COLORS.get(method, "#888")
        label = METHOD_LABELS.get(method, method)

        # Use last 500 episodes for final metrics
        for run in runs:
            final_reward = np.mean(run["rewards"][-500:])
            final_comm = np.mean(run["comm_rates"][-500:])
            ax.scatter(final_comm, final_reward, c=color, s=80, zorder=5)

        # Average point
        avg_reward = np.mean([np.mean(r["rewards"][-500:]) for r in runs])
        avg_comm = np.mean([np.mean(r["comm_rates"][-500:]) for r in runs])
        ax.scatter(
            avg_comm,
            avg_reward,
            c=color,
            s=200,
            marker="*",
            edgecolors="white",
            linewidths=1.5,
            zorder=10,
            label=label,
        )

    ax.set_xlabel("Communication Rate", fontsize=12)
    ax.set_ylabel("Average Reward (last 500 ep)", fontsize=12)
    ax.set_title("Reward vs Communication Efficiency", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate ideal corner
    ax.annotate(
        "← Better\n(less comm, more reward)",
        xy=(0.15, ax.get_ylim()[1] * 0.9),
        fontsize=9,
        color="gray",
        ha="center",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_connectivity(results: dict, save_path: str):
    """Plot connectivity loss over training (only for methods with gating)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    has_data = False
    for method, runs in results.items():
        # Skip methods with no connectivity loss
        if all(max(r["conn_losses"]) == 0 for r in runs):
            continue

        has_data = True
        color = METHOD_COLORS.get(method, "#888")
        label = METHOD_LABELS.get(method, method)

        all_conn = [smooth(np.array(r["conn_losses"])) for r in runs]
        min_len = min(len(r) for r in all_conn)
        all_conn = np.array([r[:min_len] for r in all_conn])

        mean = all_conn.mean(axis=0)
        std = all_conn.std(axis=0)

        x = np.arange(len(mean))
        ax.plot(x, mean, color=color, label=label, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    if not has_data:
        plt.close()
        return

    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Connectivity Penalty", fontsize=12)
    ax.set_title("Connectivity Penalty Over Training", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def print_summary_table(results: dict):
    """Print a summary table of final metrics."""
    print(f"\n{'Method':<25} {'Reward':>10} {'Comm Rate':>12} {'Conn Loss':>12}")
    print("-" * 62)

    for method in ["commnet", "ic3net", "tarmac", "gated_attn"]:
        if method not in results:
            continue
        runs = results[method]
        avg_reward = np.mean([np.mean(r["rewards"][-500:]) for r in runs])
        std_reward = np.std([np.mean(r["rewards"][-500:]) for r in runs])
        avg_comm = np.mean([np.mean(r["comm_rates"][-500:]) for r in runs])
        avg_conn = np.mean([np.mean(r["conn_losses"][-500:]) for r in runs])
        label = METHOD_LABELS.get(method, method)
        print(
            f"{label:<25} {avg_reward:>7.3f}±{std_reward:.3f} {avg_comm:>10.1%} {avg_conn:>12.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="paper/figures")
    parser.add_argument("--env", dest="environment", choices=ENVIRONMENTS)
    parser.add_argument("--agents", type=int)
    args = parser.parse_args()

    try:
        results = load_results(args.results_dir, args.environment, args.agents)
    except ValueError as error:
        parser.error(str(error))

    if not results:
        print(f"No results found in {args.results_dir}/")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Found methods: {list(results.keys())}")
    print(f"Runs per method: {[(k, len(v)) for k, v in results.items()]}")

    plot_rewards(results, os.path.join(args.output_dir, "rewards.png"))
    plot_comm_rates(results, os.path.join(args.output_dir, "comm_rates.png"))
    plot_pareto(results, os.path.join(args.output_dir, "pareto.png"))
    plot_connectivity(results, os.path.join(args.output_dir, "connectivity.png"))
    print_summary_table(results)


if __name__ == "__main__":
    main()
