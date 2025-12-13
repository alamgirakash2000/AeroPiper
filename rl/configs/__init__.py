"""
Task-specific training configurations.
"""

from rl.configs.base import BaseConfig
from rl.configs.pick_place import PickPlaceConfig
from rl.configs.assembly import AssemblyConfig
from rl.configs.handover import HandoverConfig

__all__ = ["BaseConfig", "PickPlaceConfig", "AssemblyConfig", "HandoverConfig"]

