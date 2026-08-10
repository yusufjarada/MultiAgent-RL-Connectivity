"""Tensor interface for MPE ``simple_spread``."""

from typing import Optional, Tuple

import numpy as np
import torch
from mpe2 import simple_spread_v3

from src.envs.base import validate_actions, validate_n_agents


class MPEWrapper:
    """Wraps simple_spread into a tensor-friendly interface."""

    def __init__(
        self, n_agents: int = 3, max_cycles: int = 25, seed: Optional[int] = None
    ):
        validate_n_agents(n_agents)
        if max_cycles < 1:
            raise ValueError("max_cycles must be positive")

        self.n_agents = n_agents
        self.max_cycles = max_cycles
        self.max_steps = max_cycles
        self._seed = seed
        self.env = simple_spread_v3.parallel_env(N=n_agents, max_cycles=max_cycles)

        # Get dimensions from a test reset
        obs, _ = self.env.reset(seed=seed)
        sample_obs = next(iter(obs.values()))
        self.obs_dim = int(sample_obs.shape[0])
        self.act_dim = int(self.env.action_space(self.env.agents[0]).n)
        self.agent_names = tuple(obs.keys())

    def reset(self, seed: Optional[int] = None) -> torch.Tensor:
        """Reset and return observations as (n_agents, obs_dim) tensor."""
        reset_seed = self._seed if seed is None else seed
        self._seed = None
        obs_dict, _ = self.env.reset(seed=reset_seed)
        obs = np.stack([obs_dict[a] for a in self.agent_names])
        return torch.as_tensor(obs, dtype=torch.float32)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, float, bool]:
        """
        Take a step with integer actions.

        Args:
            actions: (n_agents,) tensor of discrete action indices

        Returns:
            obs: (n_agents, obs_dim) tensor
            reward: scalar team reward (shared)
            done: whether episode is over
        """
        validate_actions(actions, self.n_agents, self.act_dim)
        action_list = actions.detach().cpu().numpy().tolist()
        action_dict = {
            self.agent_names[i]: action_list[i] for i in range(self.n_agents)
        }

        obs_dict, reward_dict, term_dict, trunc_dict, _ = self.env.step(action_dict)

        # Check if done
        done = all(term_dict.values()) or all(trunc_dict.values())

        reward = sum(reward_dict.values()) / self.n_agents  # shared team reward
        if done:
            # PettingZoo removes finished agents from the observation dictionary.
            # The trainer resets explicitly after consuming this transition.
            obs = torch.zeros((self.n_agents, self.obs_dim), dtype=torch.float32)
            return obs, float(reward), True

        obs = np.stack([obs_dict[a] for a in self.agent_names])
        return torch.as_tensor(obs, dtype=torch.float32), float(reward), False

    def close(self) -> None:
        self.env.close()
