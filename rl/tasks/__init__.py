"""
Task-specific RL modules.
=========================

Each task has its own module with:
- Task-specific configuration
- Reward shaping
- Curriculum logic
- Environment factory
"""

from .pick_place import PickPlaceTask

__all__ = ["PickPlaceTask"]

