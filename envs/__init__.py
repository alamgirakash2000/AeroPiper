"""
Environment registrations for Mujoco Playground.

Each environment is defined inside subpackages (e.g., manipulation).
"""

from .manipulation import (
    AeroPiperAssembly,
    AeroPiperBase,
    AeroPiperHandover,
    AeroPiperPickPlace,
)

__all__ = [
    "AeroPiperBase",
    "AeroPiperPickPlace",
    "AeroPiperHandover",
    "AeroPiperAssembly",
]
