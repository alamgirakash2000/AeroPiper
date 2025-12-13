"""
AeroPiper Gymnasium Wrappers
============================

Clean, standardized Gymnasium-style wrappers for AeroPiper manipulation environments.
These wrappers provide a consistent interface that works with any RL training framework:
- Stable-Baselines3
- RSL-RL
- CleanRL
- RLlib
- Custom implementations

Usage:
------
    from envs.wrappers import AeroPiperPickPlaceEnv, AeroPiperAssemblyEnv, AeroPiperHandoverEnv
    from envs.wrappers import make_vec_env
    
    # Single environment
    env = AeroPiperPickPlaceEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Vectorized environments for parallel training
    vec_env = make_vec_env("pick_place", num_envs=64)
"""

from envs.wrappers.base_wrapper import AeroPiperBaseEnv
from envs.wrappers.pick_place_wrapper import AeroPiperPickPlaceEnv
from envs.wrappers.pick_place_auto_grasp import AeroPiperPickPlaceAutoGrasp
from envs.wrappers.pick_place_curriculum import AeroPiperPickPlaceCurriculum
from envs.wrappers.assembly_wrapper import AeroPiperAssemblyEnv
from envs.wrappers.handover_wrapper import AeroPiperHandoverEnv
from envs.wrappers.vec_env import VecEnv, make_vec_env

__all__ = [
    "AeroPiperBaseEnv",
    "AeroPiperPickPlaceEnv",
    "AeroPiperPickPlaceAutoGrasp",
    "AeroPiperPickPlaceCurriculum",
    "AeroPiperAssemblyEnv",
    "AeroPiperHandoverEnv",
    "VecEnv",
    "make_vec_env",
]

