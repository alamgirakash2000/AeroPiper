"""
Handover Task Configuration
===========================
"""

from dataclasses import dataclass, field
from typing import List

from rl.configs.base import BaseConfig


@dataclass
class HandoverConfig(BaseConfig):
    """Configuration for handover task."""
    
    # Task-specific
    task_name: str = "handover"
    
    # Environment settings
    max_episode_steps: int = 800
    reward_type: str = "dense"
    success_threshold: float = 0.05
    randomize_objects: bool = True
    action_scale: float = 0.4
    frame_skip: int = 5
    
    # Training
    num_iterations: int = 8000
    num_envs: int = 2048
    
    # PPO tuning (coordination between arms)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.008
    
    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    # Reward weights
    reaching_weight: float = 0.4
    grasping_weight: float = 0.3
    handover_weight: float = 1.0  # Main objective
    release_weight: float = 0.5
    success_bonus: float = 15.0
    action_penalty: float = 0.01

