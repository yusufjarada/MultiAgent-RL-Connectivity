"""
Multi-agent PPO (Proximal Policy Optimization) trainer.

PPO collects a batch of experience, then runs multiple optimization
epochs on that batch with a clipped surrogate objective. This gets
much more learning per sample than REINFORCE.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PPOBuffer:
    """Stores transitions from rollouts for PPO updates."""

    def __init__(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.comm_rates = []
        self.conn_penalties = []
        self.last_value = 0.0

    def store(
        self,
        obs,
        actions,
        log_probs,
        reward,
        value,
        done,
        comm_rate=1.0,
        conn_penalty=None,
    ):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.comm_rates.append(comm_rate)
        if conn_penalty is not None:
            self.conn_penalties.append(conn_penalty)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


class ValueNetwork(nn.Module):
    """Permutation-invariant centralized critic for variable-size teams."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.agent_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            all_obs: (batch, n_agents, obs_dim)
        Returns:
            value: (batch,)
        """
        if all_obs.ndim != 3 or all_obs.shape[-1] != self.obs_dim:
            raise ValueError(
                "all_obs must have shape (batch, n_agents, "
                f"{self.obs_dim}); got {tuple(all_obs.shape)}"
            )
        if all_obs.shape[1] < 2:
            raise ValueError("the critic requires at least two agents")

        encoded = self.agent_encoder(all_obs)
        mean_pool = encoded.mean(dim=1)
        max_pool = encoded.max(dim=1).values
        team_embedding = torch.cat((mean_pool, max_pool), dim=-1)
        return self.value_head(team_embedding).squeeze(-1)


class PPOTrainer:
    def __init__(
        self,
        comm_module: nn.Module,
        env,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.comm = comm_module.to(device)
        self.env = env
        self.device = device

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        if ppo_epochs < 1 or batch_size < 1:
            raise ValueError("ppo_epochs and batch_size must be positive")

        # Centralized value network (CTDE: sees everything during training)
        self.critic = ValueNetwork(env.obs_dim).to(device)

        # Separate optimizers for actor (comm module) and critic
        self.actor_optim = optim.Adam(self.comm.parameters(), lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self._rollout_obs: Optional[torch.Tensor] = None
        self._current_ep_reward = 0.0

    def get_action_and_value(self, obs: torch.Tensor):
        """
        Forward pass: get action logits from comm module, value from critic.

        Args:
            obs: (n_agents, obs_dim)

        Returns:
            actions, log_probs, value, info
        """
        obs_batch = obs.unsqueeze(0).to(self.device)  # (1, n_agents, obs_dim)

        # All communication policies expose the same forward contract.
        logits, info = self.comm(obs_batch, hard_gate=False)

        logits = logits.squeeze(0)  # (n_agents, act_dim)

        # Sample actions from categorical distribution
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()  # (n_agents,)
        log_probs = dist.log_prob(actions).sum()  # sum across agents

        # Critic: centralized value estimate
        value = self.critic(obs_batch).squeeze(0)  # scalar

        return actions, log_probs, value, info, dist

    def compute_gae(self, rewards, values, dones, last_value: float = 0.0):
        """
        Compute Generalized Advantage Estimation (GAE).

        GAE smooths the advantage estimate between high-bias (TD)
        and high-variance (Monte Carlo) using lambda.
        """
        rewards_tensor = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        values_tensor = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        dones_tensor = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        advantages = torch.zeros_like(rewards_tensor)
        gae = torch.tensor(0.0, device=self.device)
        next_value = torch.tensor(last_value, dtype=torch.float32, device=self.device)

        for timestep in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones_tensor[timestep]
            delta = (
                rewards_tensor[timestep]
                + self.gamma * next_value * nonterminal
                - values_tensor[timestep]
            )
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            advantages[timestep] = gae
            next_value = values_tensor[timestep]

        returns = advantages + values_tensor
        return advantages, returns

    def collect_rollout(self, n_steps: int) -> Tuple[PPOBuffer, List[float]]:
        """Collect n_steps of experience across potentially multiple episodes."""
        if n_steps < 1:
            raise ValueError("n_steps must be positive")

        buf = PPOBuffer()
        if self._rollout_obs is None:
            self._rollout_obs = self.env.reset().to(self.device)

        obs = self._rollout_obs
        ep_rewards = []
        last_done = False

        for _ in range(n_steps):
            with torch.no_grad():
                actions, log_probs, value, info, _ = self.get_action_and_value(obs)

            obs_next, reward, done = self.env.step(actions)
            self._current_ep_reward += reward

            buf.store(
                obs=obs.cpu(),
                actions=actions.cpu(),
                log_probs=log_probs.cpu(),
                reward=reward,
                value=value.item(),
                done=done,
                comm_rate=info.get("comm_rate", 1.0),
                conn_penalty=info.get("conn_penalty", None),
            )

            obs = obs_next.to(self.device)
            last_done = done

            if done:
                ep_rewards.append(self._current_ep_reward)
                self._current_ep_reward = 0.0
                obs = self.env.reset().to(self.device)

        self._rollout_obs = obs
        if last_done:
            buf.last_value = 0.0
        else:
            with torch.no_grad():
                buf.last_value = float(self.critic(obs.unsqueeze(0)).squeeze(0).item())
        return buf, ep_rewards

    def update(self, buf: PPOBuffer) -> dict:
        """Run PPO update on collected buffer."""
        # Compute GAE advantages and returns
        advantages, returns = self.compute_gae(
            buf.rewards, buf.values, buf.dones, buf.last_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        # Convert buffer to tensors
        all_obs = torch.stack(buf.obs)  # (T, n_agents, obs_dim)
        all_actions = torch.stack(buf.actions)  # (T, n_agents)
        old_log_probs = torch.stack(buf.log_probs)  # (T,)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_conn_loss = 0
        n_updates = 0

        # Multiple PPO epochs over the same data
        for _epoch in range(self.ppo_epochs):
            # Shuffle indices for mini-batching
            indices = torch.randperm(len(buf))

            for start in range(0, len(buf), self.batch_size):
                end = min(start + self.batch_size, len(buf))
                idx = indices[start:end]

                batch_obs = all_obs[idx].to(self.device)
                batch_actions = all_actions[idx].to(self.device)
                batch_old_lp = old_log_probs[idx].to(self.device)
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]

                # Forward pass with current policy
                logits, info = self.comm(batch_obs, hard_gate=False)

                # New log probs and entropy
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)  # (batch,)
                entropy = dist.entropy().sum(dim=-1).mean()

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_old_lp)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                new_values = self.critic(batch_obs)
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                # Connectivity penalty
                conn_loss = torch.tensor(0.0, device=self.device)
                if "conn_penalty" in info:
                    conn_loss = info["conn_penalty"]

                # Total actor loss
                actor_loss = policy_loss - self.entropy_coef * entropy + conn_loss

                # Update actor
                self.actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.comm.parameters(), max_norm=5.0)
                self.actor_optim.step()

                # Update critic
                self.critic_optim.zero_grad()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
                self.critic_optim.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_conn_loss += conn_loss.item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "conn_loss": total_conn_loss / max(n_updates, 1),
            "avg_comm_rate": float(np.mean(buf.comm_rates)),
        }

    def train(
        self,
        total_timesteps: int = 200000,
        rollout_steps: int = 256,
        log_interval: int = 10,
        callback=None,
    ) -> Tuple[list, list]:
        """
        Train with PPO.

        Args:
            total_timesteps: total environment steps
            rollout_steps: steps per rollout before each PPO update
            log_interval: print stats every this many updates
            callback: optional function(update_num, stats)

        Returns:
            List of per-update stats.
        """
        all_stats = []
        all_ep_rewards = []
        timesteps_done = 0
        update_num = 0

        while timesteps_done < total_timesteps:
            # Collect a rollout
            steps_this_rollout = min(rollout_steps, total_timesteps - timesteps_done)
            buf, ep_rewards = self.collect_rollout(steps_this_rollout)
            timesteps_done += len(buf)
            all_ep_rewards.extend(ep_rewards)

            # PPO update
            stats = self.update(buf)
            stats["timesteps"] = timesteps_done
            stats["episodes_completed"] = len(all_ep_rewards)

            if ep_rewards:
                stats["episode_reward"] = float(np.mean(ep_rewards))
            else:
                stats["episode_reward"] = float("nan")

            all_stats.append(stats)
            update_num += 1

            if callback:
                callback(update_num, stats)

            if update_num % log_interval == 0:
                recent_rewards = all_ep_rewards[-50:] if all_ep_rewards else [0]
                print(
                    f"Update {update_num:4d} | "
                    f"Steps: {timesteps_done:7d} | "
                    f"Eps: {len(all_ep_rewards):5d} | "
                    f"Avg Reward: {np.mean(recent_rewards):7.3f} | "
                    f"Comm Rate: {stats['avg_comm_rate']:5.1%} | "
                    f"Policy Loss: {stats['policy_loss']:.4f} | "
                    f"Entropy: {stats['entropy']:.3f}"
                )

        return all_stats, all_ep_rewards
