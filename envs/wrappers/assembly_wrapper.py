"""
Assembly Environment Wrapper
============================

Gymnasium wrapper for the AeroPiper peg-in-hole assembly task.

Task Description:
-----------------
The robot must grasp a peg and insert it into a hole fixture.
This requires precise manipulation and alignment.

Observation Space:
------------------
- Robot joint positions and velocities
- End-effector positions (both arms)
- Peg position
- Target (hole) position

Action Space:
-------------
- 26-dimensional continuous actions (normalized to [-1, 1])
- Controls both arms and hands

Reward:
-------
- Dense reward based on distance to peg and peg to target
- Tighter success threshold than pick-and-place

Example:
--------
    from envs.wrappers import AeroPiperAssemblyEnv
    
    env = AeroPiperAssemblyEnv()
    obs, info = env.reset(seed=42)
    
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from envs.manipulation import AeroPiperAssembly
from envs.wrappers.base_wrapper import AeroPiperBaseEnv


class AeroPiperAssemblyEnv(AeroPiperBaseEnv):
    """
    Gymnasium environment for AeroPiper assembly task.
    
    The robot must grasp a peg and insert it into a hole.
    
    Args:
        max_episode_steps: Maximum steps before truncation (default: 500)
        reward_type: Type of reward ('dense' or 'sparse')
        success_threshold: Distance threshold for success (default: 0.015m)
        randomize_objects: Whether to randomize peg position (default: True)
        action_scale: Action scaling factor (default: 0.5)
        frame_skip: Physics steps per action (default: 5)
        render_mode: Rendering mode ('human', 'rgb_array', or None)
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        reward_type: str = "dense",
        success_threshold: float = 0.015,
        randomize_objects: bool = True,
        action_scale: float = 0.5,
        frame_skip: int = 5,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the assembly environment."""
        super().__init__(
            env_class=AeroPiperAssembly,
            max_episode_steps=max_episode_steps,
            reward_type=reward_type,
            success_threshold=success_threshold,
            randomize_objects=randomize_objects,
            action_scale=action_scale,
            frame_skip=frame_skip,
            render_mode=render_mode,
        )
        
        self.success_threshold = success_threshold
        
    def _get_info(self) -> Dict[str, Any]:
        """Get task-specific information."""
        info = super()._get_info()
        
        # Add task-specific info
        peg_pos = self._env.data.site_xpos[self._env.peg_site_id]
        target_pos = self._env.data.site_xpos[self._env.target_site_id]
        ee_pos = self.get_ee_positions()
        
        dist_to_target = np.linalg.norm(peg_pos - target_pos)
        dist_to_hand = np.linalg.norm(peg_pos - ee_pos["right"])
        
        info.update({
            "peg_pos": peg_pos.copy(),
            "target_pos": target_pos.copy(),
            "right_ee_pos": ee_pos["right"].copy(),
            "left_ee_pos": ee_pos["left"].copy(),
            "dist_peg_target": float(dist_to_target),
            "dist_peg_hand": float(dist_to_hand),
            "success": dist_to_target < self.success_threshold,
        })
        
        return info


# Convenience function for registration
def make_assembly_env(**kwargs) -> AeroPiperAssemblyEnv:
    """Factory function to create assembly environment."""
    return AeroPiperAssemblyEnv(**kwargs)

