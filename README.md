# Learned Communication with Connectivity Constraints in Multi-Agent RL

> Agents learn **when to talk**, **who to listen to**, and are **guaranteed to never lose contact** with the team.

[Interactive Demo](demo/index.html) · [Paper Figures](paper/figures/) · [Deep Dive Guide](https://github.com/yusufjarada/MultiAgent-RL-Connectivity)

---

## The Problem (No Jargon)

Imagine 8 drones searching a disaster zone. Each drone has a camera that only sees what's directly below it. To cover the whole area, they need to share what they see with each other. But radio bandwidth is limited — if every drone broadcasts everything to every other drone all the time, the channel gets flooded.

So the drones need to be smart about communication:
- **When** should I speak up? (Maybe I have nothing new to report)
- **Who** needs to hear this? (Not everyone — just the drones near me working on the same area)
- **Am I about to lose contact?** (If I go silent and my neighbor goes silent, is part of the team cut off?)

Most communication papers don't check for that last one. We do.

### A concrete example

```
Timestep 1:   Drone 0 sees a survivor → broadcasts to all → useful
Timestep 2:   Drone 0 sees the same survivor → broadcasts again → wasted bandwidth
Timestep 3:   Drone 0 learns: "broadcasting didn't help, stay quiet to save bandwidth"
              Drones 3 and 5 independently learn the same thing
              Problem: they all went quiet at the same time
              → Drones 0, 3, 5 can no longer reach each other
              → Nobody coordinated the silence — the team split apart
```

The issue is that each drone optimizes on its own. Nobody checks whether going quiet will cut off part of the team. We handle this by checking the team's connectivity before allowing a drone to go silent. If shutting off a link would split the team, that link stays open.

---

## What This Repository Contains

| What | Where | Description |
|------|-------|-------------|
| Communication modules | `src/comm/` | PyTorch implementations of 4 strategies |
| Graph math | `src/utils/graph.py` | Fiedler value, Laplacian, connectivity penalty |
| Training (PPO) | `src/training/ppo_trainer.py` | Proximal Policy Optimization for multi-agent training |
| Environment | `src/envs/mpe_wrapper.py` | MPE simple_spread wrapper |
| Training script | `scripts/train.py` | Run experiments from command line |
| Plotting | `scripts/plot_results.py` | Generate paper figures from results |
| Interactive demo | `demo/index.html` | Browser visualization of all 3 communication modes |
| Tests | `tests/` | Unit tests for graph utilities and comm modules |

---

## The Four Communication Strategies

### 1. CommNet — "Everyone yells everything, always"

Every agent broadcasts a message to every other agent at every timestep. Messages are averaged together.

```
Agent 0 → message → [average all messages] → Agent 1 uses averaged info to act
Agent 1 → message ↗
Agent 2 → message ↗
```

**Pros:** Simple, works for small teams.
**Cons:** O(N²) messages. 10 agents = 45 links. 50 agents = 1,225 links. Most messages are redundant.

📄 **Code:** [`src/comm/commnet.py`](src/comm/commnet.py)

### 2. IC3Net — "Learn when to shut up"

Same as CommNet, but each agent has a **gate** — a learned switch that decides whether to transmit. If the gate says "no," the agent's message is zeroed out.

```
Agent 0 → [GATE: ON]  → message → average → ...
Agent 1 → [GATE: OFF] → silence
Agent 2 → [GATE: ON]  → message → average → ...
```

Upside: fewer messages. Agents learn to stay quiet when they have nothing useful to say.
Downside: nothing prevents all gates from closing at once, which disconnects the team.

📄 **Code:** [`src/comm/ic3net.py`](src/comm/ic3net.py)

### 3. TarMAC — "Learn who to listen to"

Instead of averaging all messages equally, each agent uses **attention** (the same mechanism as in ChatGPT/transformers) to weight incoming messages by relevance.

```
Agent 0 asks: "Who has info I need?"
  → Agent 1's message: 80% attention weight (very relevant)
  → Agent 2's message: 15% attention weight (somewhat relevant)
  → Agent 3's message:  5% attention weight (not relevant)
```

This works better than equal averaging because not all messages are equally useful. The tradeoff: agents still communicate every timestep. There's no option to stay quiet.

📄 **Code:** [`src/comm/tarmac.py`](src/comm/tarmac.py)

### 4. Ours — "Learn when to talk AND who to listen to, with a safety net"

Combines IC3Net's gating with TarMAC's attention, and adds a **connectivity constraint** from graph theory.

```
Agent 0 → [PAIRWISE GATE] → Can I send to Agent 1? YES → attention-weighted message
                           → Can I send to Agent 2? NO  → silence
                           → Can I send to Agent 3? YES → attention-weighted message

          [CONNECTIVITY CHECK] → Is the team still connected?
                               → Fiedler value > 0.1? ✓ OK, proceed
                               → Fiedler value < 0.1? ✗ Reopen some gates!
```

**Key differences from baselines:**
- **Pairwise gating**: not just "should I talk?" but "should I talk to *you specifically*?"
- **Attention**: when I do listen, I weight messages by relevance
- **Connectivity guarantee**: the Fiedler value constraint prevents the team from ever splitting apart

📄 **Code:** [`src/comm/gated_attn.py`](src/comm/gated_attn.py)

---

## The Math

Skip this section if you just want to run experiments. If you want to know what's actually happening, read on.

### The Problem (Dec-POMDP)

We model the problem as a Decentralized Partially Observable Markov Decision Process. That's four words, each meaning something specific:
- **Decentralized** — each agent makes decisions on its own, no central controller
- **Partially Observable** — each agent only sees part of the world (its own sensor range)
- **Markov** — what happens next depends only on the current state, not the full history
- **Decision Process** — agents choose actions to maximize reward over time

Formally, it's a tuple:

```
⟨ N, S, {Aᵢ}, T, {Oᵢ}, {Ωᵢ}, R, γ ⟩
```

N agents share a world (state S), each takes actions (Aᵢ), the world changes (transition T), each agent only sees part of the world (observation Oᵢ — for a drone, that might be its position, velocity, and nearby landmarks), and the whole team gets a shared reward R, discounted over time by γ.

**The goal** in simpler terms: find policies that make the team perform as well as possible over time.

```
max  E[ Σ γᵗ R(sₜ, a₁ᵗ, ..., aₙᵗ) ]
```

### Communication Extension

Each agent also generates a message mᵢ and receives messages from neighbors. Three learned functions:

| Function | What it decides | Example |
|----------|----------------|---------|
| μᵢ(mᵢ \| oᵢ) | What to say | "I see a target at position (3, 7)" |
| gᵢⱼ(oᵢ) ∈ {0,1} | Whether to send to agent j | Gate ON → transmit, Gate OFF → stay silent |
| πᵢ(aᵢ \| oᵢ, messages) | What to do | Move left (incorporating what teammates said) |

### Quick definitions

Before we get into the graph math, some terms:
- **Node**: one agent (drone, robot, etc.)
- **Edge**: a communication link between two agents. If agent A can send to agent B, there's an edge between them.
- **Graph**: the full picture of all agents and their links — a communication network map.

### The Graph Laplacian

The communication network is a graph. The **Laplacian matrix** L encodes its structure:

```
L = D - A
```

where A is the adjacency matrix (a table of who's connected to whom — 1 means connected, 0 means not) and D is the degree matrix (how many connections each agent has, on the diagonal).

**Example** — 4 agents in a line (0—1—2—3):

```
Agent 0 connects to 1 only. Agent 1 connects to 0 and 2. And so on.

        0  1  2  3              0  1  2  3              0  1  2  3
    A = [0  1  0  0]    D = [1  0  0  0]    L = [ 1 -1  0  0]
        [1  0  1  0]        [0  2  0  0]        [-1  2 -1  0]
        [0  1  0  1]        [0  0  2  0]        [ 0 -1  2 -1]
        [0  0  1  0]        [0  0  0  1]        [ 0  0 -1  1]
```

### The Fiedler Value

The Laplacian matrix has special numbers associated with it called eigenvalues. The second-smallest eigenvalue is the **Fiedler value** (λ₂). Basically: if it's positive, the graph is connected. If it's zero, the graph is split.

```
λ₂ > 0      →  Connected (every agent can reach every other agent through some chain of links)
λ₂ ≈ 0      →  Barely hanging on, bottlenecked
λ₂ = 0      →  Split into isolated groups
λ₂ very big  →  Robustly connected (hard to break apart)
```

**Real numbers from our tests:**

| Graph | Fiedler value | What it means |
|-------|:---:|:---|
| Complete graph (everyone talks to everyone) | 8.0 | Very robust — you'd have to cut many links to disconnect |
| Line graph (chain: 0—1—2—3) | 0.586 | Connected but fragile — one cut breaks it |
| Two separate pairs (0—1, 2—3) | 0.0 | Disconnected — the pairs can't reach each other |

We set a minimum Fiedler value of 0.1 for our method. If it drops below that, the system says "we're losing connectivity" and reopens some communication links.

📄 **Code:** [`src/utils/graph.py`](src/utils/graph.py) — see `fiedler_value()` for the computation

### The Connectivity Penalty

During training, gates don't output hard 0/1 decisions — they output soft probabilities (0.7 means "70% likely to send"). This is important because gradients can flow through probabilities but not through hard switches. We build a soft adjacency matrix from these probabilities, compute the Fiedler value (using `torch.linalg.eigvalsh`, which supports backpropagation), and penalize when it drops:

```
penalty = ReLU(threshold - λ₂)
```

- If λ₂ ≥ 0.1: penalty = 0, gates are free to close
- If λ₂ < 0.1: penalty > 0, gradients push gates to reopen

This gets added to the training loss:

```
total_loss = PPO_loss + α × connectivity_penalty
```

📄 **Code:** [`src/utils/graph.py`](src/utils/graph.py) — `connectivity_penalty_torch()` function

### Training: CTDE with PPO

**Centralized Training, Decentralized Execution (CTDE)** — training and execution look different:

| Phase | What the actor sees | What the critic sees |
|-------|-------------------|---------------------|
| Training | Own observation + messages | ALL agents' observations (full picture) |
| Execution | Own observation + messages | Nothing (critic is discarded) |

Why bother? The critic can learn a much better reward prediction when it sees everything, which makes the actor's policy updates more stable. At test time, the critic is thrown away and each agent runs on its own.

**PPO (Proximal Policy Optimization)** updates the policy by:
1. Collecting a batch of experience (256 timesteps)
2. Computing advantages using GAE (how much better was each action than expected?)
3. Running 4 optimization passes over the same batch
4. Clipping the update so the policy doesn't change too drastically in one step. The ratio measures "how different is the new policy from the old one?" and the clip keeps it within a safe range:

```
ratio = π_new(a|s) / π_old(a|s)
loss = -min(ratio × advantage, clip(ratio, 1-ε, 1+ε) × advantage)
```

If the ratio drifts too far from 1.0, the clip kicks in and the gradient stops pushing.

📄 **Code:** [`src/training/ppo_trainer.py`](src/training/ppo_trainer.py)

---

## Setup

```bash
# Clone
git clone https://github.com/yusufjarada/MultiAgent-RL-Connectivity.git
cd MultiAgent-RL-Connectivity

# Create environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Run Training

```bash
# Train a single method
python scripts/train.py --method commnet --timesteps 200000 --seeds 0 1 2

# Train all methods (takes ~30 min)
python scripts/train.py --method all --timesteps 200000 --seeds 0 1 2

# Options
#   --method:   commnet | ic3net | tarmac | gated_attn | all
#   --agents:   number of agents (default: 3)
#   --timesteps: total environment steps (default: 200000)
#   --seeds:    random seeds for reproducibility
```

## Generate Figures

```bash
python scripts/plot_results.py
# Outputs: paper/figures/rewards.png, comm_rates.png, pareto.png, connectivity.png
```

## Run Tests

```bash
python tests/test_graph.py
python tests/test_comm_modules.py
python tests/test_demo_logic.py
```

## Interactive Demos

Two browser demos are included — open either HTML file directly, no server needed.

### Coverage Demo (`demo/index.html`)
Agents spread out to cover target positions. Toggle between communication modes and adjust agent count / comm range with sliders. Hover over agents to see their radio range.

### Multi-Target Pursuit Demo (`demo/pursuit.html`)
The more interesting demo. N agents (4-16) must capture M moving targets (1-5) by surrounding them. This requires the agents to:
1. **Discover** targets (limited sight range)
2. **Share** target locations through the communication network
3. **Split** into subgroups — decide who goes after which target
4. **Surround** each target from multiple angles

This is where communication mode matters most. With broadcast, agents coordinate a clean split. With gated communication, subgroups lose contact and accidentally double up on the same target. With our method, the connectivity constraint keeps the relay chain between subgroups alive, so the team splits efficiently.

The demo tracks **cumulative bandwidth** — the total number of messages sent over time. Broadcast uses 100% of possible bandwidth. Our method achieves similar capture rates at 40-60% bandwidth.

---

## Project Structure

```
marl-comms/
├── src/
│   ├── comm/
│   │   ├── commnet.py       # Baseline: broadcast + mean-pool
│   │   ├── ic3net.py        # Baseline: learned per-agent gate
│   │   ├── tarmac.py        # Baseline: attention-based communication
│   │   └── gated_attn.py    # OURS: pairwise gating + attention + Fiedler constraint
│   ├── envs/
│   │   └── mpe_wrapper.py   # MPE simple_spread environment wrapper
│   ├── training/
│   │   └── ppo_trainer.py    # PPO trainer (used for all experiments)
│   └── utils/
│       └── graph.py          # Graph Laplacian, Fiedler value, connectivity penalty
├── scripts/
│   ├── train.py              # Training script (CLI)
│   └── plot_results.py       # Result visualization
├── demo/
│   ├── index.html            # Interactive browser demo
│   ├── main.js               # Demo simulation logic
│   └── style.css             # Styling
├── tests/
│   ├── test_graph.py         # Graph utility tests
│   ├── test_comm_modules.py  # Communication module tests
│   └── test_demo_logic.py    # Demo logic verification
├── paper/
│   └── figures/              # Generated plots
├── results/                  # Training results (JSON + model weights)
├── requirements.txt
└── README.md
```

---

## Environment: MPE Simple Spread

We use the [Multi-Agent Particle Environment](https://pettingzoo.farama.org/environments/mpe/) (MPE) `simple_spread` task:

- **N agents, N landmarks** on a 2D field
- **Goal:** agents must spread out to cover all landmarks
- **Reward:** negative distance from agents to landmarks (closer = less negative = better)
- **Observations:** each agent sees its own position/velocity and relative positions of landmarks and other agents (18-dimensional vector for 3 agents)
- **Actions:** discrete — up, down, left, right, stay

This task requires coordination: if all agents go to the same landmark, they get a bad reward. Communication helps agents say "I'm heading to landmark A, you take B."

---

## References

- Sukhbaatar et al., 2016 — [CommNet: Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736)
- Foerster et al., 2016 — [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676)
- Das et al., 2019 — [TarMAC: Targeted Multi-Agent Communication](https://arxiv.org/abs/1810.11187)
- Singh et al., 2019 — [IC3Net: Learning When to Communicate at Scale](https://arxiv.org/abs/1812.09755)
- Oliehoek & Amato, 2016 — [A Concise Introduction to Decentralized POMDPs](https://link.springer.com/book/10.1007/978-3-319-28929-8)
- Schulman et al., 2017 — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

---

## Authors

**Yusuf Jarada** — Robotics Engineering, Purdue University

Course project for AAE 590: Multi-Agent Autonomy and Control (Prof. Shaoshuai Mou)

---

## License

MIT
