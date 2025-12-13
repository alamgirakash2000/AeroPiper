"""
Base Training Configuration
===========================
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class BaseConfig:
    """Base configuration for all tasks."""
    
    # Environment
    num_envs: int = 128
    max_episode_steps: int = 2000
    
    # Training
    num_iterations: int = 5000
    num_steps_per_env: int = 64
    
    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    
    # Learning rate
    learning_rate: float = 3e-4
    lr_schedule: str = "adaptive"
    
    # Mini-batch
    num_mini_batches: int = 4
    num_epochs: int = 5
    
    # Network
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"
    init_noise_std: float = 1.0
    
    # Logging
    log_interval: int = 10
    save_interval: int = 500
    
    # Device
    device: str = "cuda"
    
    # Wandb
    wandb_project: str = "aeropiper-rl"

