# Research Roadmap and Publication Plan

Last literature search: 2026-08-09

## Research objective

Develop a scalable multi-agent reinforcement-learning method in which agents learn a task-conditioned, sparse communication graph while respecting physical communication availability and an explicit algebraic-connectivity requirement. Evaluate the tradeoff among task performance, bandwidth, and connectivity violations in physics-backed multi-robot tasks.

## Chosen paper direction

**Working title:** Robust Connectivity-Constrained Communication Learning for Embodied Multi-Robot Teams

The paper will study joint motion and communication under unreliable, range-limited links. At each step, the motion policy changes the physically available graph, while the communication policy selects a sparse subgraph for message passing. A constrained learner and an execution-time safety layer will keep the selected graph above a stated algebraic-connectivity threshold whenever the physical graph makes that feasible.

The intended result is:

> A learned policy can approach the task performance and communication efficiency of unconstrained sparse communication while providing enforceable pre-failure connectivity under changing robot geometry and improved robustness to subsequent link failures.

This is the result we will attempt to establish. It is not yet a supported claim.

### Primary contributions

1. A constrained MARL formulation for jointly learning robot motion and sparse communication over a time-varying physical-link graph.
2. A minimal-intervention safety projection that repairs proposed communication graphs when needed and explicitly identifies states in which the physical graph makes the constraint infeasible.
3. Robust training and evaluation under stochastic and distance-dependent packet loss, with separate reporting of proposed-graph, repaired-graph, and executed-graph connectivity.
4. A systematic three-way characterization of task performance, communication cost, and connectivity robustness across team sizes.

### Supporting, not standalone, contributions

- The variable-agent MuJoCo platform is experimental infrastructure, not the paper's novelty.
- Pairwise gating and attention are established components.
- The existing differentiable Fiedler penalty is a baseline and initialization point, not the final method.
- Scaling to more agents supports the claims but is not sufficient as a contribution by itself.

### Scope control

The first paper will not initially claim learned adaptive thresholds, mutual-information optimality, arbitrary communication delays, or a general theorem for every MARL architecture. These remain follow-on directions. We will add one only if the primary method is working and the extension directly strengthens the central claim.

## Research questions and falsifiable hypotheses

### RQ1: Can constrained learning produce sparse graphs that satisfy connectivity without relying constantly on a safety layer?

**H1:** Primal-dual training plus safety projection will reduce communication relative to full communication, have materially fewer violations than a soft penalty, and require progressively fewer projection interventions during training.

### RQ2: Does enforcing connectivity help task performance under unreliable links?

**H2:** Under packet loss and changing geometry, the constrained method will have higher success and lower catastrophic disconnection rates than unconstrained sparse communication at matched communication cost.

### RQ3: Does joint motion adapt to communication constraints?

**H3:** Compared with communication-only graph selection, joint motion and communication will create relay-like behavior or connectivity-preserving formations when direct task-optimal motion would fragment the physical graph.

### RQ4: Does the approach scale and generalize?

**H4:** A shared permutation-equivariant policy trained on selected team sizes will retain useful performance on unseen team sizes, with computation and communication scaling reported explicitly.

A result is scientifically useful even if a hypothesis fails. For example, if `lambda_2` does not predict robustness or task performance, the paper can establish when algebraic connectivity is an inadequate proxy. We must preserve negative results rather than tune them away.

## Formal problem definition

At time `t`, define four distinct graphs:

- `G_phys(t)`: feasible directed or undirected links determined by robot geometry and radio assumptions.
- `G_prop(t)`: sparse links proposed by the learned communication policy, constrained to be a subgraph of `G_phys(t)`.
- `G_safe(t)`: the proposed graph after any feasible safety repair but before stochastic link failures.
- `G_exec(t)`: links actually available after stochastic link failures affect `G_safe(t)`.

The main optimization target is

`maximize  E[return] - beta * E[communication_cost]`

subject to an explicitly chosen constraint, initially

`lambda_2(L_safe(t)) >= lambda_min`

whenever a satisfying subgraph exists inside `G_phys(t)`. Robustness of `G_exec(t)` under failures will first be treated empirically and through risk-sensitive training. A chance constraint such as

`P(lambda_2(L_exec(t)) >= lambda_min) >= 1 - delta`

will be claimed only if the algorithm and analysis actually support it.

Communication cost will be defined as directed transmissions per step, with message bytes and distance-weighted energy reported when applicable. Environment reward and communication cost must remain separate metrics even if both enter the learning objective.

## Proposed method

### Learned proposal policy

- Shared agent encoder and permutation-equivariant pairwise edge scores.
- Physical availability masking before edge sampling.
- Discrete or straight-through edge selection for an executable sparse graph.
- Message aggregation only across executed links.
- Shared actor with a permutation-invariant centralized critic during training.

### Primal-dual constrained learning

Replace a hand-tuned fixed penalty with a learned nonnegative Lagrange multiplier that responds to observed constraint violations. Retain the current fixed soft penalty as a baseline.

The implementation must log the task loss, communication cost, constraint value, multiplier, and gradient behavior independently. Stabilization choices such as multiplier clipping or slower dual updates must be documented rather than hidden.

### Minimal-intervention safety projection

After the policy proposes `G_prop`, the safety layer will:

1. Test whether `G_phys` itself can satisfy the specified connectivity requirement.
2. If infeasible, report a physical-graph infeasibility event rather than attributing it to the policy.
3. If feasible and `G_prop` violates the requirement, add feasible edges using a deterministic minimal-edge or greedy spectral-gain procedure until the threshold is reached.
4. Record every added edge, projection invocation, and residual violation.

We will distinguish a proven exact minimum from a heuristic repair. Unless optimality is established, call the method a greedy or approximate minimal-intervention projection.

### Robustness model

Begin with independent and distance-dependent packet loss. Then add one structured failure condition, such as a temporary regional outage or a single-link failure. Train with sampled failures and compare expected-risk and tail-risk objectives if time permits.

The safety guarantee, if any, applies to the graph before random post-projection packet loss unless a robust projection or chance-constrained result is established. This boundary must be explicit in the paper.

## Research phases and decision gates

### Phase 0: Measurement and reproducibility foundation

- Freeze the current MuJoCo environment as a versioned baseline.
- Add deterministic experiment configuration and structured per-step graph logging.
- Validate `lambda_2`, graph masks, edge counts, and failures against hand-constructed graphs.
- Produce a small reference experiment that can be reproduced from one command.

**Gate:** Do not develop the new optimizer until graph metrics and replayed runs agree exactly for fixed seeds.

### Phase 1: Physical communication model

- Separate physical availability, proposed communication, and executed communication in the environment/API.
- Add configurable radio range and packet-loss models.
- Create scenarios in which direct target pursuit can disconnect the physical graph and relay behavior is useful.
- Add an explicit infeasibility signal when `G_phys` cannot meet the threshold.

**Gate:** Hand-authored policies and graph fixtures must demonstrate feasible, repairable, and physically infeasible cases.

### Phase 2: Constraint enforcement

- Implement soft-penalty, primal-dual, and safety-projection variants behind one clean interface.
- Verify the projection exhaustively on all small undirected graphs up to a practical agent count.
- Establish precisely what is guaranteed and under which graph, symmetry, threshold, and failure assumptions.

**Gate:** The safety layer must produce zero avoidable pre-failure violations in exhaustive small-graph tests before being described as enforcing connectivity.

### Phase 3: Joint learning

- Train the communication policy and motion policy together.
- Test whether the constrained learner reduces intervention rate without sacrificing success.
- Diagnose collapse modes: full communication, no communication, dual-variable explosion, oscillating topology, and motion policies that exploit the physical mask.

**Gate:** Continue to the full study only if the method beats both full communication on bandwidth and unconstrained sparsity on connectivity at a comparable task-success level.

### Phase 4: Robustness, scaling, and generalization

- Run multiple team sizes and unseen-size transfer.
- Sweep range, packet-loss rate, communication budget, and `lambda_min`.
- Add at least one structured link-failure test.
- Profile training/inference time and graph-operation scaling.

**Gate:** The principal conclusion must hold across seeds and more than one environment configuration, not only one favorable operating point.

### Phase 5: Analysis and paper production

- Lock the method before final benchmark runs.
- Register the planned comparisons, seeds, and primary metrics in the repository.
- Generate every table and figure from immutable raw result files.
- Perform a fresh literature audit and claim-to-evidence review.
- Release code, configs, checkpoints, and a result manifest tied to the paper commit.

## Immediate implementation backlog

These are the next tasks, in order:

1. Write the graph semantics specification for `G_phys`, `G_prop`, `G_safe`, and `G_exec`, including directed-versus-undirected behavior.
2. Refactor the environment and communication interface so these graphs cannot be confused.
3. Add range-limited and stochastic link models with deterministic tests.
4. Add structured graph telemetry and an evaluation command that reports all primary safety and bandwidth metrics.
5. Implement and exhaustively test the safety projection on small graphs.
6. Implement primal-dual constraint training and compare it with the existing soft penalty.
7. Design one relay-demanding MuJoCo task before undertaking large-scale training.

## Primary evaluation metrics

- Task return, success rate, completion time, and collision rate.
- Directed transmissions, active-edge fraction, bytes, and distance-weighted communication energy.
- `lambda_2` for physical, proposed, safe, and executed graphs.
- Avoidable policy violations versus physical-graph infeasibility events.
- Violation probability, duration, severity, and disconnected-episode rate.
- Safety-projection invocation rate, edges added, and intervention trend over training.
- Robustness curves against packet loss and communication range.
- Generalization across seen and unseen team sizes.
- Wall-clock time, parameter count, and graph computation cost.

## Current novelty assessment

Novelty is not yet established. No exact duplicate was found in the searches recorded below, but several papers cover major pieces of the proposed method. In particular, Connectivity-Driven Communication (CDC) is close because it learns a state-dependent weighted communication graph and uses Laplacian spectral structure. A paper must distinguish itself experimentally and technically from this work, not merely from broadcast communication.

The current implementation uses a soft penalty

`alpha * ReLU(lambda_min - lambda_2(L_soft))`.

This encourages algebraic connectivity. It is not a mathematical guarantee that every executed communication graph satisfies the threshold. Do not use the word "guarantee" unless the method gains a proof, a safety filter/projection, or another mechanism that enforces the constraint under clearly stated assumptions.

## Closest prior work

| Work | Existing contribution | Distinction that may remain for this project |
| --- | --- | --- |
| [Connectivity-Driven Communication (Machine Learning, 2023)](https://link.springer.com/article/10.1007/s10994-022-06286-6) | Learns a state-dependent, undirected weighted communication graph; uses graph diffusion and Laplacian spectral structure to control information flow. | Explicit constrained optimization of `lambda_2`, physical availability masks, violation guarantees/accounting, and task-bandwidth-connectivity Pareto analysis may distinguish the project. |
| [Learning Multi-Agent Communication from Graph Modeling Perspective (ICLR 2024)](https://openreview.net/forum?id=Qox9rO0kN0) | Treats inter-agent communication architecture as a learnable graph rather than a fixed topology. | The contribution cannot be merely "learn the graph"; it must center on connectivity-constrained graph selection in embodied teams. |
| [CGIBNet (2021)](https://arxiv.org/abs/2112.10374) | Learns compact messages and graph structure through graph information bottlenecks. | Explicit physical connectivity and algebraic-connectivity control may remain distinct. |
| [IMAC (ICML 2020)](https://proceedings.mlr.press/v119/wang20i.html) | Jointly learns informative messages and a scheduler under limited bandwidth, including whom to contact. | The differentiator must be connectivity-constrained topology, not communication efficiency alone. |
| [TarMAC (ICML 2019)](https://proceedings.mlr.press/v97/das19a.html) | Learns targeted messages and recipients through attention. | Pairwise targeting and attention are established components, not novel claims. |
| [Learning Structured Communication (2020)](https://arxiv.org/abs/2002.04235) | Learns adaptive hierarchical communication topology for scalable MARL. | Scaling and learned topology alone are insufficient novelty. |
| [Learning Practical Communication Strategies (AIML 2023)](https://proceedings.mlr.press/v189/hu23a.html) | Studies practical MARL communication under realistic communication constraints. | The paper needs direct comparisons on bandwidth and robustness, not only return. |
| [Connectivity Guaranteed Multi-Robot Navigation via Deep RL (CoRL 2020)](https://proceedings.mlr.press/v100/lin20a.html) | Uses deep RL for multi-robot navigation while maintaining network connectivity. | This project learns the communication topology itself and targets bandwidth-task tradeoffs, but must compare its form of connectivity enforcement. |
| [Communication-Aware Trajectory Planning by Constraining the Fiedler Value (2024)](https://arxiv.org/abs/2406.18452) | Directly constrains the Fiedler value in multi-robot trajectory optimization and demonstrates simulation and real-robot results. | RL-based learned communication rather than offline trajectory optimization may distinguish the work; this is important precedent for any Fiedler-constraint claim. |
| [Nonsmooth Control Barrier Functions for Network Connectivity (2021)](https://arxiv.org/abs/2112.05935) | Provides control-theoretic connectivity maintenance using barrier functions. | A safety layer based on this family could support a real guarantee; a penalty-only method should be contrasted honestly. |

## Paper experiment standard

- Multiple tasks and team sizes, including at least one physics-backed multi-robot task.
- At least 5 seeds, confidence intervals, learning curves, final-policy evaluation, and compute reporting.
- Baselines: no communication, full/broadcast communication, TarMAC-style targeting, IMAC or an information-bottleneck method, a learnable-graph method such as CDC or the ICLR 2024 graph model, and relevant connectivity-preserving control/RL.
- Ablations: no connectivity term, no sparsity cost, different thresholds, penalty versus constrained optimization, physical mask on/off, and attention/gating choices.
- Metrics: return/success, communication rate and volume, `lambda_2` distribution, violation frequency and duration, disconnected episodes, robustness, and scaling cost.
- Pareto plots showing task performance versus communication and connectivity violations.
- Reproducible configs, environment versions, seeds, raw metrics, checkpoints, and scripts that regenerate tables and figures.

## Publication logistics

- Decide authorship and contributions early; discuss scope and claims with the faculty advisor.
- Choose a target venue before locking the experiment scale and format. arXiv is a public preprint host, not peer review.
- Prepare a self-contained LaTeX manuscript, bibliography, legible vector figures, appendix/supplement, and an anonymous version if the venue requires it.
- Confirm arXiv account/endorsement, category, author approval, licenses, and source-package requirements.
- Archive the exact code commit, configs, environment lock file, and result data used by the paper.
- Run a final claim-to-evidence audit and a fresh literature search before submission.

Likely arXiv categories are `cs.MA` as primary, with `cs.RO` or `cs.LG` as possible cross-lists depending on the final emphasis.

## Search limitations

This is a living review, not proof that no similar paper exists. Search indexes miss papers, terminology varies, and new work appears continuously. Before submission, search Google Scholar, Semantic Scholar, IEEE Xplore, ACM Digital Library, OpenReview, arXiv, and relevant robotics/control proceedings; follow citations both backward and forward from the closest papers.
