"""
Base Gymnasium Wrapper for AeroPiper Environments
==================================================

This module provides a base class that wraps AeroPiper manipulation environments
with the standard Gymnasium API (https://gymnasium.farama.org/).

The wrapper handles:
- Gymnasium-compliant observation and action spaces
- Standard reset() and step() signatures
- Proper termination vs truncation handling
- Rendering support (human, rgb_array modes)
- Seeding for reproducibility

Example:
--------
    class MyCustomEnv(AeroPiperBaseEnv):
        def __init__(self):
            super().__init__(
                env_class=AeroPiperPickPlace,
                max_episode_steps=500,
            )
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class AeroPiperBaseEnv(gym.Env, ABC):
    """
    Base Gymnasium wrapper for AeroPiper manipulation environments.
    
    This class provides a standardized Gymnasium interface for all AeroPiper tasks.
    Subclasses should implement task-specific observation processing and info dict.
    
    Attributes:
        observation_space: Gymnasium Box space for observations
        action_space: Gymnasium Box space for actions
        metadata: Environment metadata including render modes and fps
        
    Args:
        env_class: The underlying AeroPiper environment class
        max_episode_steps: Maximum steps per episode before truncation
        action_scale: Scaling factor for actions (default from env)
        frame_skip: Number of physics steps per action (default from env)
        render_mode: Rendering mode ('human', 'rgb_array', or None)
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        env_class: type,
        max_episode_steps: int = 500,
        action_scale: Optional[float] = None,
        frame_skip: Optional[int] = None,
        render_mode: Optional[str] = None,
        **env_kwargs,
    ) -> None:
        """Initialize the Gymnasium wrapper."""
        super().__init__()
        
        # Store configuration
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0
        
        # Build kwargs for underlying environment
        self._env_kwargs = env_kwargs.copy()
        if action_scale is not None:
            self._env_kwargs["action_scale"] = action_scale
        if frame_skip is not None:
            self._env_kwargs["frame_skip"] = frame_skip
            
        # Create underlying environment
        self._env_class = env_class
        self._env = env_class(**self._env_kwargs)
        
        # Define action space (normalized to [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
            dtype=np.float32,
        )
        
        # Define observation space (determined from a sample observation)
        sample_obs = self._env.reset()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )
        
        # Store references for convenience
        self.model = self._env.model
        self.data = self._env.data
        
    @property
    def unwrapped(self) -> "AeroPiperBaseEnv":
        """Return the unwrapped environment."""
        return self
    
    @property
    def np_random(self) -> np.random.Generator:
        """Return the random number generator."""
        if not hasattr(self, "_np_random"):
            self._np_random = np.random.default_rng()
        return self._np_random
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused)
            
        Returns:
            observation: Initial observation as numpy array
            info: Dictionary with additional information
        """
        # Handle seeding
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
            
        # Reset underlying environment
        obs = self._env.reset(rng=self._np_random)
        self._step_count = 0
        
        # Build info dict
        info = self._get_info()
        
        return obs.astype(np.float32), info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: Action to execute (normalized to [-1, 1])
            
        Returns:
            observation: New observation after action
            reward: Scalar reward value
            terminated: Whether episode ended due to task completion/failure
            truncated: Whether episode ended due to time limit
            info: Dictionary with additional information
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        
        # Step underlying environment
        obs, reward, done, env_info = self._env.step(action)
        self._step_count += 1
        
        # Determine termination vs truncation
        truncated = self._step_count >= self.max_episode_steps
        terminated = done and not truncated
        
        # Build info dict
        info = self._get_info()
        info.update(env_info)
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.
        
        Returns:
            For 'rgb_array' mode: RGB image as numpy array
            For 'human' mode: None (displays in window)
        """
        if self.render_mode == "rgb_array":
            return self._env.render(width=640, height=480)
        elif self.render_mode == "human":
            # For human mode, we'd need to set up a viewer
            # This is handled by the evaluation script's viewer
            return self._env.render(width=640, height=480)
        return None
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass  # MuJoCo handles cleanup automatically
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get current environment info dictionary.
        
        Override in subclasses to add task-specific information.
        
        Returns:
            Dictionary with environment state information
        """
        return {
            "step_count": self._step_count,
            "max_episode_steps": self.max_episode_steps,
        }
    
    # ================================================================
    # Properties for accessing underlying environment
    # ================================================================
    
    @property
    def action_size(self) -> int:
        """Number of action dimensions."""
        return self._env.action_size
    
    @property
    def frame_skip(self) -> int:
        """Number of physics steps per action."""
        return self._env.frame_skip
    
    @property
    def action_scale(self) -> float:
        """Action scaling factor."""
        return self._env.action_scale
    
    def get_ee_positions(self) -> Dict[str, np.ndarray]:
        """Get end-effector positions."""
        return self._env.get_ee_positions()
    
    def get_fingertip_positions(self) -> Dict[str, np.ndarray]:
        """Get fingertip positions."""
        return self._env.get_fingertip_positions()

