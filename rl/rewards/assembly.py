"""
Assembly Reward Function
========================

Reward shaping for the assembly task.
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class AssemblyRewardConfig:
    """Configuration for assembly reward."""
    reaching_weight: float = 0.3
    alignment_weight: float = 0.5
    insertion_weight: float = 1.0
    success_bonus: float = 20.0
    action_penalty: float = 0.005


class AssemblyReward:
    """
    Assembly reward function with shaping.
    
    Reward components:
    1. Reaching: Encourage hand to approach peg
    2. Alignment: Reward for correct orientation
    3. Insertion: Main reward for insertion progress
    4. Success: Large bonus for task completion
    5. Action smoothness: Penalty for jerky movements
    """
    
    def __init__(self, config: Optional[AssemblyRewardConfig] = None):
        self.config = config or AssemblyRewardConfig()
        self.prev_actions = None
        
    def __call__(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        next_obs: torch.Tensor,
        base_reward: torch.Tensor,
        done: torch.Tensor,
        info: list,
    ) -> torch.Tensor:
        """Compute shaped reward for assembly task."""
        device = obs.device
        batch_size = obs.shape[0]
        
        reward = base_reward.clone()
        
        if isinstance(info, list) and len(info) > 0:
            for i, inf in enumerate(info):
                # Success bonus
                if inf.get("success", False):
                    reward[i] += self.config.success_bonus
                
                # Alignment bonus (if tracked)
                alignment_error = inf.get("alignment_error", 1.0)
                if alignment_error < 0.1:  # Good alignment
                    reward[i] += self.config.alignment_weight * (0.1 - alignment_error)
        
        # Action smoothness
        if self.prev_actions is not None:
            action_diff = torch.sum((action - self.prev_actions) ** 2, dim=-1)
            reward -= self.config.action_penalty * action_diff
        
        self.prev_actions = action.clone()
        
        return reward
    
    def reset(self):
        """Reset reward state."""
        self.prev_actions = None


def create_assembly_reward(config_dict: dict = None) -> AssemblyReward:
    """Factory function to create assembly reward."""
    if config_dict:
        config = AssemblyRewardConfig(**config_dict)
    else:
        config = AssemblyRewardConfig()
    return AssemblyReward(config)

