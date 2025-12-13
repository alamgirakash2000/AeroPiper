"""
RL Module for AeroPiper
=======================

Task-specific RL training code including:
- PPO algorithm implementation
- Task-specific reward functions
- Training configurations
- Vectorized environments
"""

from rl.ppo import PPO, PPOConfig
from rl.networks import ActorCritic
from rl.vec_env import VecEnv

__all__ = ["PPO", "PPOConfig", "ActorCritic", "VecEnv"]

