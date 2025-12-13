"""
Handover Environment Wrapper
============================

Gymnasium wrapper for the AeroPiper bimanual handover task.

Task Description:
-----------------
The robot must pick up a ball with one hand and transfer it to the other hand,
then place it at a target location. This requires coordination between both arms.

Observation Space:
------------------
- Robot joint positions and velocities
- End-effector positions (both arms)
- Ball position
- Target position

Action Space:
-------------
- 26-dimensional continuous actions (normalized to [-1, 1])
- Controls both arms and hands simultaneously

Reward:
-------
- Dense reward considering distances to both hands
- Encourages proper handover sequence

Example:
--------
    from envs.wrappers import AeroPiperHandoverEnv
    
    env = AeroPiperHandoverEnv()
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

from envs.manipulation import AeroPiperHandover
from envs.wrappers.base_wrapper import AeroPiperBaseEnv


class AeroPiperHandoverEnv(AeroPiperBaseEnv):
    """
    Gymnasium environment for AeroPiper handover task.
    
    The robot must transfer a ball between hands and place it at a target.
    
    Args:
        max_episode_steps: Maximum steps before truncation (default: 500)
        reward_type: Type of reward ('dense' or 'sparse')
        success_threshold: Distance threshold for success (default: 0.05m)
        randomize_objects: Whether to randomize ball position (default: True)
        action_scale: Action scaling factor (default: 0.5)
        frame_skip: Physics steps per action (default: 5)
        render_mode: Rendering mode ('human', 'rgb_array', or None)
    """
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        reward_type: str = "dense",
        success_threshold: float = 0.05,
        randomize_objects: bool = True,
        action_scale: float = 0.5,
        frame_skip: int = 5,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the handover environment."""
        super().__init__(
            env_class=AeroPiperHandover,
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
        ball_pos = self._env.data.site_xpos[self._env.ball_site_id]
        target_pos = self._env.data.site_xpos[self._env.target_site_id]
        ee_pos = self.get_ee_positions()
        
        dist_to_target = np.linalg.norm(ball_pos - target_pos)
        dist_to_right = np.linalg.norm(ball_pos - ee_pos["right"])
        dist_to_left = np.linalg.norm(ball_pos - ee_pos["left"])
        
        info.update({
            "ball_pos": ball_pos.copy(),
            "target_pos": target_pos.copy(),
            "right_ee_pos": ee_pos["right"].copy(),
            "left_ee_pos": ee_pos["left"].copy(),
            "dist_ball_target": float(dist_to_target),
            "dist_ball_right": float(dist_to_right),
            "dist_ball_left": float(dist_to_left),
            "success": dist_to_target < self.success_threshold,
        })
        
        return info


# Convenience function for registration
def make_handover_env(**kwargs) -> AeroPiperHandoverEnv:
    """Factory function to create handover environment."""
    return AeroPiperHandoverEnv(**kwargs)

