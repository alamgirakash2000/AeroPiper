"""
Pick-and-Place Reward Function
==============================

Reward shaping for the pick-and-place task.
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class PickPlaceRewardConfig:
    """Configuration for pick-place reward."""
    reaching_weight: float = 0.5
    grasping_weight: float = 0.3
    lifting_weight: float = 0.3
    placing_weight: float = 1.0
    success_bonus: float = 10.0
    action_penalty: float = 0.01


class PickPlaceReward:
    """
    Pick-and-place reward function with shaping.
    
    Reward components:
    1. Reaching: Encourage hand to approach cube
    2. Grasping: Reward when cube is close to hand
    3. Lifting: Reward for lifting cube above table
    4. Placing: Main reward for cube approaching target
    5. Success: Large bonus for task completion
    6. Action smoothness: Penalty for large actions
    """
    
    def __init__(self, config: Optional[PickPlaceRewardConfig] = None):
        self.config = config or PickPlaceRewardConfig()
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
        """
        Compute shaped reward.
        
        This function wraps around the base environment reward and adds
        additional shaping terms.
        
        Args:
            obs: Current observation
            action: Action taken
            next_obs: Next observation
            base_reward: Reward from base environment
            done: Done flags
            info: Info dicts from environments
            
        Returns:
            Shaped reward tensor
        """
        device = obs.device
        batch_size = obs.shape[0]
        
        # Start with base reward (already includes distance terms)
        reward = base_reward.clone()
        
        # Extract info from environments (if available)
        if isinstance(info, list) and len(info) > 0:
            # Add success bonus
            for i, inf in enumerate(info):
                if inf.get("success", False):
                    reward[i] += self.config.success_bonus
                    
                # Add grasping bonus based on distance
                dist_hand = inf.get("dist_cube_hand", 1.0)
                if dist_hand < 0.05:  # Within 5cm
                    reward[i] += self.config.grasping_weight
        
        # Action smoothness penalty
        if self.prev_actions is not None:
            action_diff = torch.sum((action - self.prev_actions) ** 2, dim=-1)
            reward -= self.config.action_penalty * action_diff
        
        self.prev_actions = action.clone()
        
        return reward
    
    def reset(self):
        """Reset reward state (call on episode reset)."""
        self.prev_actions = None


def create_pick_place_reward(config_dict: dict = None) -> PickPlaceReward:
    """Factory function to create pick-place reward."""
    if config_dict:
        config = PickPlaceRewardConfig(**config_dict)
    else:
        config = PickPlaceRewardConfig()
    return PickPlaceReward(config)

