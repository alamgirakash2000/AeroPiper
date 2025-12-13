"""
Handover Reward Function
========================

Reward shaping for the handover task.
"""

import torch
import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class HandoverRewardConfig:
    """Configuration for handover reward."""
    reaching_weight: float = 0.4
    grasping_weight: float = 0.3
    handover_weight: float = 1.0
    release_weight: float = 0.5
    success_bonus: float = 15.0
    action_penalty: float = 0.01


class HandoverReward:
    """
    Handover reward function with shaping.
    
    Reward components:
    1. Reaching: First hand approaches object
    2. Grasping: First hand secures object
    3. Handover: Move object toward second hand
    4. Release: Second hand takes object
    5. Success: Large bonus for successful handover
    6. Action smoothness: Penalty for jerky movements
    """
    
    def __init__(self, config: Optional[HandoverRewardConfig] = None):
        self.config = config or HandoverRewardConfig()
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
        """Compute shaped reward for handover task."""
        device = obs.device
        batch_size = obs.shape[0]
        
        reward = base_reward.clone()
        
        if isinstance(info, list) and len(info) > 0:
            for i, inf in enumerate(info):
                # Success bonus
                if inf.get("success", False):
                    reward[i] += self.config.success_bonus
                
                # Handover progress (object moving toward second hand)
                handover_progress = inf.get("handover_progress", 0.0)
                reward[i] += self.config.handover_weight * handover_progress
                
                # Release bonus when second hand has object
                if inf.get("object_in_second_hand", False):
                    reward[i] += self.config.release_weight
        
        # Action smoothness
        if self.prev_actions is not None:
            action_diff = torch.sum((action - self.prev_actions) ** 2, dim=-1)
            reward -= self.config.action_penalty * action_diff
        
        self.prev_actions = action.clone()
        
        return reward
    
    def reset(self):
        """Reset reward state."""
        self.prev_actions = None


def create_handover_reward(config_dict: dict = None) -> HandoverReward:
    """Factory function to create handover reward."""
    if config_dict:
        config = HandoverRewardConfig(**config_dict)
    else:
        config = HandoverRewardConfig()
    return HandoverReward(config)

