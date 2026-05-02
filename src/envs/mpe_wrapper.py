"""
Wrapper for MPE (Multi-Agent Particle Environment) that converts
PettingZoo's dict-based API into batched tensors for our comm modules.
"""

import numpy as np
import torch
from mpe2 import simple_spread_v3


class MPEWrapper:
    """Wraps simple_spread into a tensor-friendly interface."""

    def __init__(self, n_agents: int = 3, max_cycles: int = 25):
        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=max_cycles)

        # Get dimensions from a test reset
        obs, _ = self.env.reset()
        sample_obs = next(iter(obs.values()))
        self.obs_dim = sample_obs.shape[0]
        self.act_dim = self.env.action_space(self.env.agents[0]).n
        self.agent_names = list(obs.keys())

    def reset(self) -> torch.Tensor:
        """Reset and return observations as (n_agents, obs_dim) tensor."""
        obs_dict, _ = self.env.reset()
        obs = np.stack([obs_dict[a] for a in self.agent_names])
        return torch.FloatTensor(obs)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, float, bool]:
        """
        Take a step with integer actions.

        Args:
            actions: (n_agents,) tensor of discrete action indices

        Returns:
            obs: (n_agents, obs_dim) tensor
            reward: scalar team reward (shared)
            done: whether episode is over
        """
        action_list = actions.cpu().numpy().tolist()
        action_dict = {self.agent_names[i]: action_list[i] for i in range(self.n_agents)}

        obs_dict, reward_dict, term_dict, trunc_dict, _ = self.env.step(action_dict)

        # Check if done
        done = all(term_dict.values()) or all(trunc_dict.values())

        if done:
            obs = self.reset()
            reward = sum(reward_dict.values()) / self.n_agents
            return obs, reward, True

        obs = np.stack([obs_dict[a] for a in self.agent_names])
        reward = sum(reward_dict.values()) / self.n_agents  # shared team reward

        return torch.FloatTensor(obs), reward, False

    def close(self):
        self.env.close()
