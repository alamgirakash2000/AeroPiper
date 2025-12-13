"""
Compatibility wrapper for vectorized environments.

`scripts/train.py` expects `VecEnv(env_fn, num_envs, device=...)` while the
shared implementation in `envs.wrappers.vec_env` takes a list of env factory
functions. This module adapts the signature and re-exports the helpers.
"""

from envs.wrappers.vec_env import (  # type: ignore
    VecEnv as _BaseVecEnv,
    make_vec_env,
    make_pick_place_vec_env,
    make_assembly_vec_env,
    make_handover_vec_env,
)


class VecEnv(_BaseVecEnv):
    """Adapter around the base VecEnv to accept (env_fn, num_envs, device)."""

    def __init__(self, env_fn, num_envs: int = 1, device: str = "cuda"):
        super().__init__([env_fn for _ in range(num_envs)], device=device)


__all__ = [
    "VecEnv",
    "make_vec_env",
    "make_pick_place_vec_env",
    "make_assembly_vec_env",
    "make_handover_vec_env",
]

