"""
Vectorized Environment Wrapper
==============================

Provides parallel environment execution for efficient RL training.
Compatible with both synchronous and asynchronous execution patterns.

This module provides:
- VecEnv: Base vectorized environment class
- make_vec_env: Factory function to create vectorized environments

Usage:
------
    from envs.wrappers import make_vec_env
    
    # Create 64 parallel environments
    vec_env = make_vec_env("pick_place", num_envs=64, device="cuda")
    
    # Reset all environments
    obs = vec_env.reset()
    
    # Step all environments
    obs, rewards, dones, infos = vec_env.step(actions)
    
    # Close when done
    vec_env.close()
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from envs.wrappers.pick_place_wrapper import AeroPiperPickPlaceEnv
from envs.wrappers.assembly_wrapper import AeroPiperAssemblyEnv
from envs.wrappers.handover_wrapper import AeroPiperHandoverEnv


# Registry of available environments
ENV_REGISTRY = {
    "pick_place": AeroPiperPickPlaceEnv,
    "assembly": AeroPiperAssemblyEnv,
    "handover": AeroPiperHandoverEnv,
}


class VecEnv:
    """
    Vectorized environment for parallel training.
    
    Runs multiple environment instances in parallel and batches
    observations, rewards, and dones into tensors for efficient
    GPU-based training.
    
    Attributes:
        num_envs: Number of parallel environments
        observation_space: Gymnasium observation space
        action_space: Gymnasium action space
        obs_dim: Observation dimension
        action_dim: Action dimension
        device: Torch device for tensors
        
    Args:
        env_fns: List of callables that create environments
        device: Device for tensor operations ('cuda' or 'cpu')
    """
    
    def __init__(
        self,
        env_fns: List[Callable[[], Any]],
        device: str = "cuda",
    ) -> None:
        """Initialize vectorized environment."""
        self.num_envs = len(env_fns)
        self.device = device
        
        # Create all environments
        self.envs = [fn() for fn in env_fns]
        
        # Get spaces from first environment
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Cache dimensions
        self.obs_dim = self.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]
        
        # Initialize buffers
        self._obs_buffer = torch.zeros(
            (self.num_envs, self.obs_dim),
            dtype=torch.float32,
            device=device,
        )
        self._reward_buffer = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=device,
        )
        self._done_buffer = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=device,
        )
        self._truncated_buffer = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=device,
        )
        
    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Reset all environments.
        
        Args:
            seed: Base seed (each env gets seed + env_idx)
            
        Returns:
            observations: Batched observations tensor [num_envs, obs_dim]
            infos: List of info dicts from each environment
        """
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            self._obs_buffer[i] = torch.from_numpy(obs).to(self.device)
            infos.append(info)
            
        return self._obs_buffer.clone(), infos
    
    def step(
        self,
        actions: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Step all environments.
        
        Args:
            actions: Batched actions [num_envs, action_dim]
            
        Returns:
            observations: New observations [num_envs, obs_dim]
            rewards: Rewards [num_envs]
            terminated: Termination flags [num_envs]
            truncated: Truncation flags [num_envs]
            infos: List of info dicts
        """
        # Convert actions to numpy if needed
        if isinstance(actions, torch.Tensor):
            actions_np = actions.cpu().numpy()
        else:
            actions_np = actions
            
        infos = []
        
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions_np[i])
            
            self._obs_buffer[i] = torch.from_numpy(obs).to(self.device)
            self._reward_buffer[i] = reward
            self._done_buffer[i] = terminated
            self._truncated_buffer[i] = truncated
            infos.append(info)
            
            # Auto-reset on done
            if terminated or truncated:
                # IMPORTANT: Save terminal info BEFORE reset overwrites it
                # These are the values from the completed episode
                terminal_success_reach = info.get("success_reach", False)
                terminal_success_place = info.get("success_place", False)
                terminal_success = info.get("success", False)
                terminal_ep_reach = info.get("episode_reach_success", False)
                terminal_ep_place = info.get("episode_place_success", False)
                
                obs, reset_info = env.reset()
                self._obs_buffer[i] = torch.from_numpy(obs).to(self.device)
                info["terminal_observation"] = obs
                info.update(reset_info)
                
                # Restore terminal values (these are what we care about for tracking)
                info["success_reach"] = terminal_success_reach
                info["success_place"] = terminal_success_place
                info["success"] = terminal_success
                info["episode_reach_success"] = terminal_ep_reach
                info["episode_place_success"] = terminal_ep_place
                
        return (
            self._obs_buffer.clone(),
            self._reward_buffer.clone(),
            self._done_buffer.clone(),
            self._truncated_buffer.clone(),
            infos,
        )
    
    def step_simple(
        self,
        actions: Union[torch.Tensor, np.ndarray],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simplified step that returns combined done signal.
        
        Useful for simpler training loops that don't need
        to distinguish between termination and truncation.
        
        Args:
            actions: Batched actions [num_envs, action_dim]
            
        Returns:
            observations: New observations [num_envs, obs_dim]
            rewards: Rewards [num_envs]
            dones: Combined done flags [num_envs]
        """
        obs, rewards, terminated, truncated, _ = self.step(actions)
        dones = terminated | truncated
        return obs, rewards, dones
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
            
    def render(self) -> Optional[np.ndarray]:
        """Render first environment (for visualization)."""
        return self.envs[0].render()
    
    @property
    def unwrapped(self) -> "VecEnv":
        """Return the unwrapped environment."""
        return self
    
    def get_attr(self, attr_name: str) -> List[Any]:
        """Get attribute from all environments."""
        return [getattr(env, attr_name) for env in self.envs]
    
    def set_attr(self, attr_name: str, value: Any) -> None:
        """Set attribute on all environments."""
        for env in self.envs:
            setattr(env, attr_name, value)
            
    def env_method(self, method_name: str, *args, **kwargs) -> List[Any]:
        """Call method on all environments."""
        return [getattr(env, method_name)(*args, **kwargs) for env in self.envs]


def make_vec_env(
    env_id: str,
    num_envs: int = 1,
    device: str = "cuda",
    seed: Optional[int] = None,
    **env_kwargs,
) -> VecEnv:
    """
    Create a vectorized environment.
    
    Factory function that creates multiple environment instances
    wrapped in a VecEnv for parallel training.
    
    Args:
        env_id: Environment identifier ('pick_place', 'assembly', 'handover')
        num_envs: Number of parallel environments
        device: Device for tensor operations
        seed: Random seed (each env gets seed + idx)
        **env_kwargs: Additional arguments passed to environment constructor
        
    Returns:
        VecEnv instance with num_envs parallel environments
        
    Example:
        vec_env = make_vec_env("pick_place", num_envs=64, device="cuda")
        obs, infos = vec_env.reset(seed=42)
    """
    if env_id not in ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment: {env_id}. "
            f"Available: {list(ENV_REGISTRY.keys())}"
        )
    
    env_class = ENV_REGISTRY[env_id]
    
    def make_env(idx: int) -> Callable:
        def _init() -> Any:
            env = env_class(**env_kwargs)
            return env
        return _init
    
    env_fns = [make_env(i) for i in range(num_envs)]
    
    return VecEnv(env_fns, device=device)


# Convenience aliases
def make_pick_place_vec_env(num_envs: int = 64, **kwargs) -> VecEnv:
    """Create vectorized pick-and-place environment."""
    return make_vec_env("pick_place", num_envs=num_envs, **kwargs)


def make_assembly_vec_env(num_envs: int = 64, **kwargs) -> VecEnv:
    """Create vectorized assembly environment."""
    return make_vec_env("assembly", num_envs=num_envs, **kwargs)


def make_handover_vec_env(num_envs: int = 64, **kwargs) -> VecEnv:
    """Create vectorized handover environment."""
    return make_vec_env("handover", num_envs=num_envs, **kwargs)

