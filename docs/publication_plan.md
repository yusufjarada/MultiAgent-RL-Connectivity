# Publication and Prior-Work Plan

Last literature search: 2026-08-09

## Research objective

Develop a scalable multi-agent reinforcement-learning method in which agents learn a task-conditioned, sparse communication graph while respecting physical communication availability and an explicit algebraic-connectivity requirement. Evaluate the tradeoff among task performance, bandwidth, and connectivity violations in physics-backed multi-robot tasks.

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

## Provisional contribution worth testing

The strongest plausible contribution is:

> A constrained MARL formulation that learns a sparse, task-conditioned communication graph over physically available robot links, minimizing communication while keeping algebraic-connectivity violations below a stated tolerance, with systematic task-performance, bandwidth, and connectivity tradeoff evaluation.

This is a hypothesis for a contribution, not a confirmed novelty claim.

## Required technical work

1. Replace or augment the soft penalty with a principled constrained method, such as a Lagrangian/primal-dual update, and report threshold violations explicitly.
2. If claiming a hard guarantee, add an enforceable projection or safety-filter mechanism and state its assumptions. Otherwise use "connectivity regularization" or "empirical constraint satisfaction."
3. Define physical-link availability separately from the learned communication decision. An edge outside radio/sensing range must not be selectable.
4. Define communication cost precisely: directed transmissions, active edges, bytes, or messages per step.
5. Establish scaling across agent counts and evaluate unseen team sizes.
6. Add meaningful disturbances: packet loss, limited range, delays if supported, moving obstacles, and robot dynamics.

## Minimum experimental package

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
