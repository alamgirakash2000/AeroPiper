"""
Pick-and-Place Task Configuration
=================================

Configuration for curriculum-based pick-and-place training.

Key changes from standard config:
- 6D action space: [arm_select, j1, j2, j3, j4, j5], joint6=0
- Curriculum learning: reach first, then place
- Gradual progress: memory of best control points
- Fixed positions until success achieved
"""

from dataclasses import dataclass, field
from typing import List

from rl.configs.base import BaseConfig


@dataclass
class PickPlaceConfig(BaseConfig):
    """Configuration for pick-and-place task with curriculum learning."""
    
    # Task-specific
    task_name: str = "pick_place"
    
    # Environment settings
    max_episode_steps: int = 500  # Shorter episodes, terminate on success
    reward_type: str = "dense"
    success_threshold: float = 0.10  # 10cm for success
    randomize_objects: bool = False  # Start fixed, enable with --randomize
    action_scale: float = 0.3  # Lower for more gradual movement
    frame_skip: int = 5
    disable_curriculum: bool = True  # Default: train reach+place together
    
    # Training
    num_iterations: int = 10000  # Can cancel anytime
    num_envs: int = 64  # More envs = faster training
    num_steps_per_env: int = 128  # Longer rollouts for multi-step tasks
    
    # PPO tuning for gradual learning
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.005  # Low entropy for focused exploration
    max_grad_norm: float = 1.0
    
    # Learning rate
    learning_rate: float = 1e-4  # Good default for pick-place
    lr_schedule: str = "adaptive"
    
    # Mini-batch settings
    num_mini_batches: int = 4
    num_epochs: int = 5
    
    # Network (moderate size)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 0.5  # Lower noise for more deterministic actions
    
    # Curriculum settings
    phase_transition_threshold: float = 0.80  # 80% reach success to unlock place
    randomize_after_streak: int = 5  # Consecutive successes before randomizing
    distance_tolerance: float = 0.002  # 2mm for progress detection
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100  # Save checkpoint every 100 iterations

