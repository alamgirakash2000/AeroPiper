"""
Pick-and-Place Environment Wrapper
===================================

Gymnasium wrapper for the AeroPiper pick-and-place manipulation task.

Task Description:
-----------------
The robot must pick up a cube from a random position on the table
and place it at a target location.

Observation Space:
------------------
- Robot joint positions and velocities
- End-effector positions (both arms)
- Cube position
- Target position

Action Space:
-------------
- 26-dimensional continuous actions (normalized to [-1, 1])
- Controls both arms and hands

Reward:
-------
- Dense reward based on distance to cube and cube to target
- Negative reward proportional to distances

Example:
--------
    from envs.wrappers import AeroPiperPickPlaceEnv
    
    env = AeroPiperPickPlaceEnv()
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

from envs.manipulation import AeroPiperPickPlace
from envs.wrappers.base_wrapper import AeroPiperBaseEnv


class AeroPiperPickPlaceEnv(AeroPiperBaseEnv):
    """
    Gymnasium environment for AeroPiper pick-and-place task.
    
    The robot must pick up a cube and place it at a target location.
    
    Args:
        max_episode_steps: Maximum steps before truncation (default: 500)
        reward_type: Type of reward ('dense' or 'sparse')
        success_threshold: Distance threshold for success (default: 0.04m)
        randomize_objects: Whether to randomize cube position (default: True)
        action_scale: Action scaling factor (default: 0.5)
        frame_skip: Physics steps per action (default: 5)
        render_mode: Rendering mode ('human', 'rgb_array', or None)
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        reward_type: str = "dense",
        success_threshold: float = 0.04,
        randomize_objects: bool = True,
        action_scale: float = 0.5,
        frame_skip: int = 5,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the pick-and-place environment."""
        super().__init__(
            env_class=AeroPiperPickPlace,
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
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id]
        target_pos = self._env.data.site_xpos[self._env.target_site_id]
        ee_pos = self.get_ee_positions()
        
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        dist_to_hand = np.linalg.norm(cube_pos - ee_pos["right"])
        
        info.update({
            "cube_pos": cube_pos.copy(),
            "target_pos": target_pos.copy(),
            "right_ee_pos": ee_pos["right"].copy(),
            "left_ee_pos": ee_pos["left"].copy(),
            "dist_cube_target": float(dist_to_target),
            "dist_cube_hand": float(dist_to_hand),
            "success": dist_to_target < self.success_threshold,
        })
        
        return info


# Convenience function for registration
def make_pick_place_env(**kwargs) -> AeroPiperPickPlaceEnv:
    """Factory function to create pick-and-place environment."""
    return AeroPiperPickPlaceEnv(**kwargs)

