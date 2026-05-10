# CLAUDE.md — Agent Context for MultiAgent-RL-Connectivity

## What this repo is

A research project for AAE 590 (Multi-Agent Autonomy and Control) at Purdue, taught by Prof. Shaoshuai Mou. The topic is learned communication in multi-agent reinforcement learning (MARL) with a connectivity constraint from algebraic graph theory.

The one-sentence pitch: agents learn when to talk and who to listen to, with a mathematical guarantee (Fiedler value) that the team never loses connectivity.

## Who

Yusuf Jarada — Robotics Engineering, Purdue University. Email: yjarada@purdue.edu. GitHub: yusufjarada.

## Repo structure

```
src/
  comm/           — 4 communication modules (PyTorch)
    commnet.py    — Baseline: broadcast + mean-pool
    ic3net.py     — Baseline: learned per-agent gate
    tarmac.py     — Baseline: multi-head attention
    gated_attn.py — OURS: pairwise gating + attention + Fiedler constraint
  envs/
    mpe_wrapper.py — MPE simple_spread environment wrapper
  training/
    ppo_trainer.py — PPO with centralized critic (CTDE)
  utils/
    graph.py       — Fiedler value, Laplacian, differentiable connectivity penalty
scripts/
  train.py              — Train all methods (CLI)
  plot_results.py       — Generate paper figures from results
  export_model_to_json.py — Export PyTorch weights to JSON for browser demo
  export_trajectories.py  — Record trained policy episodes for replay
demo/
  index.html     — Coverage demo (simulated communication modes)
  pursuit.html   — Multi-target pursuit demo (the impressive one)
  pursuit.js     — Pursuit simulation logic
  nn.js          — Neural net inference in pure JavaScript
  models.json    — Exported model weights for live browser inference
  trajectories.json — Recorded episodes from trained policies
paper/
  main.tex       — Final report (IEEE format, 4 pages)
  references.bib — BibTeX references
  figures/       — Generated plots (rewards, comm_rates, pareto)
tests/
  test_graph.py        — Graph utility tests
  test_comm_modules.py — Communication module tests
  test_demo_logic.py   — Demo logic verification
```

## How to run

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Train (all 4 methods, 5 seeds, 500k timesteps)
python scripts/train.py --method all --timesteps 500000 --seeds 0 1 2 3 4

# Generate figures
python scripts/plot_results.py

# Run tests
python tests/test_graph.py && python tests/test_comm_modules.py

# Compile paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Open demos
open demo/pursuit.html
open demo/index.html
```

## Key technical details

- **Environment**: MPE simple_spread (3 agents, 18-dim obs, 5 discrete actions). Installed via `mpe2` package.
- **Training**: PPO with centralized critic (CTDE). 256-step rollouts, 4 PPO epochs, GAE lambda 0.95, clip 0.2.
- **Connectivity penalty**: `ReLU(0.1 - λ₂(L_soft))` where λ₂ is the Fiedler value computed via `torch.linalg.eigvalsh` on the soft gate Laplacian. Weight α = 0.5.
- **Models are tiny**: 10k-22k params. Train in minutes on CPU/MPS.
- **Results** (500k steps, 5 seeds): CommNet -20.7 @ 100% comm, Ours -21.6 @ 44% comm. 56% bandwidth savings for 4% reward cost.

## Current status

- Midterm report submitted (in Downloads/MidtermReport/)
- Final report written and submitted (paper/main.tex, also at ~/Desktop/FinalReport.pdf)
- All training complete (500k steps, 5 seeds, 4 methods)
- Two interactive demos working (coverage + pursuit)
- Live neural net inference in browser working
- Repo pushed to github.com/yusufjarada/MultiAgent-RL-Connectivity

## What's next (if continuing this research)

1. **Scale to more agents** (8, 12, 20) — the main selling point is that broadcast doesn't scale. Need to prove it empirically.
2. **Add communication cost to reward** — penalize each message so bandwidth savings show up in the reward signal.
3. **Curriculum on connectivity weight α** — start at 0, ramp up during training.
4. **Consensus-based aggregation** — replace attention with classical consensus iterations (ties to Prof. Mou's work).
5. **Train on pursuit task** — currently the pursuit demo is simulated, not trained. Training agents on multi-target pursuit with MARL would be a strong result.

## Style notes

- All academic writing must be humanized (run /humanizer skill). No em dashes, no formulaic enumerations, no AI vocabulary.
- Yusuf understands both RL and classical control theory. No need to simplify either side.
- Prof. Mou cares about analytical results, not just empirical benchmarks.

## Git

- Remote: github.com/yusufjarada/MultiAgent-RL-Connectivity
- Auth: `yjarada03` has collaborator access. Push with: `GITHUB_TOKEN=$GITHUB_TOKEN git push https://yjarada03:$GITHUB_TOKEN@github.com/yusufjarada/MultiAgent-RL-Connectivity.git main`
- Repo-level git config: user.name=yusufjarada, user.email=yjarada@purdue.edu
