"""
Assembly Task Configuration
===========================
"""

from dataclasses import dataclass, field
from typing import List

from rl.configs.base import BaseConfig


@dataclass
class AssemblyConfig(BaseConfig):
    """Configuration for assembly task."""
    
    # Task-specific
    task_name: str = "assembly"
    
    # Environment settings
    max_episode_steps: int = 1000  # Longer for complex task
    reward_type: str = "dense"
    success_threshold: float = 0.02  # Tighter tolerance for assembly
    randomize_objects: bool = True
    action_scale: float = 0.3  # Slower, more precise
    frame_skip: int = 5
    
    # Training (more iterations for harder task)
    num_iterations: int = 10000
    num_envs: int = 2048
    
    # PPO tuning
    gamma: float = 0.995  # Higher for long-horizon
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01  # More exploration for complex task
    
    # Network (larger for complex task)
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    # Reward weights
    reaching_weight: float = 0.3
    alignment_weight: float = 0.5  # Important for assembly
    insertion_weight: float = 1.0
    success_bonus: float = 20.0
    action_penalty: float = 0.005

