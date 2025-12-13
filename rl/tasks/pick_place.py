"""
Pick-and-Place Task Module
==========================

Task-specific training logic for pick-and-place:
- Curriculum learning (reach first, then place)
- Gradual learning with memory of best control points
- Progress tracking and metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class PickPlaceCurriculum:
    """Curriculum state for pick-place training."""
    
    # Current phase: "reach" or "place"
    current_phase: str = "reach"
    
    # Success tracking
    reach_successes: List[float] = field(default_factory=list)
    place_successes: List[float] = field(default_factory=list)
    
    # Thresholds
    reach_success_threshold: float = 0.10  # 10cm
    place_success_threshold: float = 0.10  # 10cm
    phase_transition_rate: float = 0.80  # 80% reach success to unlock place
    phase_window: int = 1000  # Last N steps to average
    
    # Success streak for randomization
    success_streak: int = 0
    randomize_threshold: int = 10  # Randomize after N consecutive successes
    
    # Best achieved distances
    best_reach_dist: float = float('inf')
    best_place_dist: float = float('inf')
    
    def update(self, reach_success: bool, place_success: bool, 
               reach_dist: float, place_dist: float) -> Dict[str, Any]:
        """Update curriculum state and return metrics."""
        self.reach_successes.append(float(reach_success))
        self.place_successes.append(float(place_success))
        
        # Update best distances
        if reach_dist < self.best_reach_dist:
            self.best_reach_dist = reach_dist
        if place_dist < self.best_place_dist:
            self.best_place_dist = place_dist
        
        # Update success streak
        full_success = reach_success and place_success
        if full_success:
            self.success_streak += 1
        else:
            self.success_streak = 0
        
        # Calculate recent success rates
        recent_reach = self.reach_successes[-self.phase_window:]
        recent_place = self.place_successes[-self.phase_window:]
        reach_rate = np.mean(recent_reach) if recent_reach else 0.0
        place_rate = np.mean(recent_place) if recent_place else 0.0
        
        # Phase transition logic
        old_phase = self.current_phase
        if self.current_phase == "reach":
            if reach_rate >= self.phase_transition_rate:
                self.current_phase = "place"
        
        return {
            "phase": self.current_phase,
            "phase_changed": old_phase != self.current_phase,
            "reach_rate": reach_rate,
            "place_rate": place_rate,
            "best_reach_dist": self.best_reach_dist,
            "best_place_dist": self.best_place_dist,
            "success_streak": self.success_streak,
            "should_randomize": self.success_streak >= self.randomize_threshold,
        }
    
    def is_reach_only(self) -> bool:
        """Whether to focus only on reaching."""
        return self.current_phase == "reach"


@dataclass  
class GradualLearningMemory:
    """
    Memory for gradual learning approach.
    
    Tracks control points that brought the robot closer to the target.
    Allows reverting to previous good positions when exploration fails.
    """
    
    # Best control values per phase
    best_reach_ctrl: Optional[np.ndarray] = None
    best_place_ctrl: Optional[np.ndarray] = None
    
    # Best distances achieved
    best_reach_dist: float = float('inf')
    best_place_dist: float = float('inf')
    
    # Previous step values for comparison
    prev_reach_dist: float = float('inf')
    prev_place_dist: float = float('inf')
    prev_ctrl: Optional[np.ndarray] = None
    
    # Improvement history
    reach_improvements: List[float] = field(default_factory=list)
    place_improvements: List[float] = field(default_factory=list)
    
    # Revert counter
    reverts_count: int = 0
    steps_since_improvement: int = 0
    max_steps_without_improvement: int = 50
    
    def reset(self, initial_ctrl: np.ndarray, initial_reach_dist: float, 
              initial_place_dist: float) -> None:
        """Reset memory at episode start."""
        self.best_reach_ctrl = initial_ctrl.copy()
        self.best_place_ctrl = initial_ctrl.copy()
        self.best_reach_dist = initial_reach_dist
        self.best_place_dist = initial_place_dist
        self.prev_reach_dist = initial_reach_dist
        self.prev_place_dist = initial_place_dist
        self.prev_ctrl = initial_ctrl.copy()
        self.reach_improvements.clear()
        self.place_improvements.clear()
        self.reverts_count = 0
        self.steps_since_improvement = 0
    
    def update_reach(self, new_dist: float, new_ctrl: np.ndarray, 
                     tolerance: float = 0.001) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Update reach phase memory.
        
        Returns:
            (improved, revert_ctrl): Whether improved and ctrl to revert to if needed
        """
        improvement = self.prev_reach_dist - new_dist
        self.reach_improvements.append(improvement)
        
        if new_dist < self.best_reach_dist - tolerance:
            # New best! Save this control point
            self.best_reach_dist = new_dist
            self.best_reach_ctrl = new_ctrl.copy()
            self.steps_since_improvement = 0
            self.prev_reach_dist = new_dist
            self.prev_ctrl = new_ctrl.copy()
            return True, None
        elif new_dist > self.prev_reach_dist + tolerance:
            # Got worse - should revert
            self.reverts_count += 1
            self.steps_since_improvement += 1
            # Return the best ctrl to revert to
            return False, self.best_reach_ctrl
        else:
            # Roughly same - continue exploring
            self.steps_since_improvement += 1
            self.prev_reach_dist = new_dist
            self.prev_ctrl = new_ctrl.copy()
            return False, None
    
    def update_place(self, new_dist: float, new_ctrl: np.ndarray,
                     tolerance: float = 0.001) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Update place phase memory.
        
        Returns:
            (improved, revert_ctrl): Whether improved and ctrl to revert to if needed
        """
        improvement = self.prev_place_dist - new_dist
        self.place_improvements.append(improvement)
        
        if new_dist < self.best_place_dist - tolerance:
            # New best! Save this control point
            self.best_place_dist = new_dist
            self.best_place_ctrl = new_ctrl.copy()
            self.steps_since_improvement = 0
            self.prev_place_dist = new_dist
            self.prev_ctrl = new_ctrl.copy()
            return True, None
        elif new_dist > self.prev_place_dist + tolerance:
            # Got worse - should revert
            self.reverts_count += 1
            self.steps_since_improvement += 1
            return False, self.best_place_ctrl
        else:
            # Roughly same - continue exploring
            self.steps_since_improvement += 1
            self.prev_place_dist = new_dist
            self.prev_ctrl = new_ctrl.copy()
            return False, None
    
    def should_explore_new_direction(self) -> bool:
        """Whether stuck and should try a different direction."""
        return self.steps_since_improvement > self.max_steps_without_improvement
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "best_reach_dist": self.best_reach_dist,
            "best_place_dist": self.best_place_dist,
            "reverts_count": self.reverts_count,
            "steps_since_improvement": self.steps_since_improvement,
            "avg_reach_improvement": np.mean(self.reach_improvements[-100:]) if self.reach_improvements else 0,
            "avg_place_improvement": np.mean(self.place_improvements[-100:]) if self.place_improvements else 0,
        }


class PickPlaceRewardShaper:
    """
    Reward shaper for pick-place task with curriculum support.
    
    Provides dense rewards based on:
    1. Distance reduction (main signal)
    2. Phase-specific bonuses
    3. Improvement bonuses
    4. Penalties for moving away
    """
    
    def __init__(
        self,
        reach_bonus: float = 1.0,
        place_bonus: float = 5.0,
        improvement_scale: float = 10.0,
        regression_penalty: float = 0.5,
        success_bonus_reach: float = 5.0,
        success_bonus_place: float = 20.0,
    ):
        self.reach_bonus = reach_bonus
        self.place_bonus = place_bonus
        self.improvement_scale = improvement_scale
        self.regression_penalty = regression_penalty
        self.success_bonus_reach = success_bonus_reach
        self.success_bonus_place = success_bonus_place
    
    def compute(
        self,
        phase: str,
        prev_dist: float,
        new_dist: float,
        reach_success: bool,
        place_success: bool,
        base_reward: float = 0.0,
        improved: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward.
        
        Returns:
            (total_reward, reward_components)
        """
        components = {}
        
        # Distance change reward (main signal)
        dist_improvement = prev_dist - new_dist
        components["dist_improvement"] = dist_improvement * self.improvement_scale
        
        # Regression penalty
        if dist_improvement < -0.001:
            components["regression_penalty"] = -self.regression_penalty
        else:
            components["regression_penalty"] = 0.0
        
        # Phase-specific distance penalty (encourage getting closer)
        if phase == "reach":
            components["distance_penalty"] = -new_dist * 0.5
        else:
            components["distance_penalty"] = -new_dist * 1.0
        
        # Success bonuses
        if reach_success and phase == "reach":
            components["reach_success"] = self.success_bonus_reach
        else:
            components["reach_success"] = 0.0
        
        if place_success:
            components["place_success"] = self.success_bonus_place
        else:
            components["place_success"] = 0.0
        
        # Improvement bonus
        if improved:
            components["improvement_bonus"] = 0.5
        else:
            components["improvement_bonus"] = 0.0
        
        # Base reward passthrough
        components["base"] = base_reward
        
        # Total
        total = sum(components.values())
        return total, components


class PickPlaceTask:
    """
    Task-specific training manager for pick-and-place.
    
    Handles:
    - Curriculum state management
    - Gradual learning memory
    - Reward shaping
    - Progress tracking and logging
    """
    
    def __init__(
        self,
        reach_threshold: float = 0.10,
        place_threshold: float = 0.10,
        phase_transition_rate: float = 0.80,
        randomize_after_streak: int = 10,
    ):
        self.curriculum = PickPlaceCurriculum(
            reach_success_threshold=reach_threshold,
            place_success_threshold=place_threshold,
            phase_transition_rate=phase_transition_rate,
            randomize_threshold=randomize_after_streak,
        )
        self.reward_shaper = PickPlaceRewardShaper()
        
        # Per-environment memory
        self.env_memories: Dict[int, GradualLearningMemory] = {}
        
        # Global progress tracking
        self.total_reach_successes = 0
        self.total_place_successes = 0
        self.total_episodes = 0
        
        # Best ever achieved
        self.global_best_reach = float('inf')
        self.global_best_place = float('inf')
        
        # Recent episode stats
        self.episode_reach_dists: List[float] = []
        self.episode_place_dists: List[float] = []
    
    def get_memory(self, env_idx: int) -> GradualLearningMemory:
        """Get or create memory for an environment."""
        if env_idx not in self.env_memories:
            self.env_memories[env_idx] = GradualLearningMemory()
        return self.env_memories[env_idx]
    
    def on_episode_start(
        self,
        env_idx: int,
        initial_ctrl: np.ndarray,
        reach_dist: float,
        place_dist: float,
    ) -> None:
        """Called when an episode starts."""
        memory = self.get_memory(env_idx)
        memory.reset(initial_ctrl, reach_dist, place_dist)
    
    def on_step(
        self,
        env_idx: int,
        reach_dist: float,
        place_dist: float,
        new_ctrl: np.ndarray,
        base_reward: float,
    ) -> Tuple[float, bool, Optional[np.ndarray], Dict[str, Any]]:
        """
        Process a step for gradual learning.
        
        Returns:
            (shaped_reward, improved, revert_ctrl, info)
        """
        memory = self.get_memory(env_idx)
        phase = self.curriculum.current_phase
        
        # Determine which distance to track based on phase
        if phase == "reach":
            improved, revert_ctrl = memory.update_reach(reach_dist, new_ctrl)
            prev_dist = memory.prev_reach_dist if hasattr(memory, 'prev_reach_dist') else reach_dist
            current_dist = reach_dist
        else:
            improved, revert_ctrl = memory.update_place(place_dist, new_ctrl)
            prev_dist = memory.prev_place_dist if hasattr(memory, 'prev_place_dist') else place_dist
            current_dist = place_dist
        
        # Check successes
        reach_success = reach_dist < self.curriculum.reach_success_threshold
        place_success = place_dist < self.curriculum.place_success_threshold
        
        # Compute shaped reward
        shaped_reward, reward_components = self.reward_shaper.compute(
            phase=phase,
            prev_dist=prev_dist,
            new_dist=current_dist,
            reach_success=reach_success,
            place_success=place_success,
            base_reward=base_reward,
            improved=improved,
        )
        
        info = {
            "phase": phase,
            "improved": improved,
            "should_revert": revert_ctrl is not None,
            "reach_success": reach_success,
            "place_success": place_success,
            "reward_components": reward_components,
            **memory.get_stats(),
        }
        
        return shaped_reward, improved, revert_ctrl, info
    
    def on_episode_end(
        self,
        env_idx: int,
        final_reach_dist: float,
        final_place_dist: float,
        success: bool,
    ) -> Dict[str, Any]:
        """Called when an episode ends."""
        self.total_episodes += 1
        self.episode_reach_dists.append(final_reach_dist)
        self.episode_place_dists.append(final_place_dist)
        
        # Update curriculum
        reach_success = final_reach_dist < self.curriculum.reach_success_threshold
        place_success = final_place_dist < self.curriculum.place_success_threshold
        
        if reach_success:
            self.total_reach_successes += 1
        if place_success:
            self.total_place_successes += 1
        
        # Update global bests
        if final_reach_dist < self.global_best_reach:
            self.global_best_reach = final_reach_dist
        if final_place_dist < self.global_best_place:
            self.global_best_place = final_place_dist
        
        curriculum_info = self.curriculum.update(
            reach_success, place_success,
            final_reach_dist, final_place_dist
        )
        
        return {
            "episode_reach_dist": final_reach_dist,
            "episode_place_dist": final_place_dist,
            "reach_success": reach_success,
            "place_success": place_success,
            **curriculum_info,
        }
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current training progress summary."""
        reach_rate = self.total_reach_successes / max(1, self.total_episodes)
        place_rate = self.total_place_successes / max(1, self.total_episodes)
        
        recent_reach = self.episode_reach_dists[-100:]
        recent_place = self.episode_place_dists[-100:]
        
        return {
            "total_episodes": self.total_episodes,
            "reach_success_rate": reach_rate,
            "place_success_rate": place_rate,
            "current_phase": self.curriculum.current_phase,
            "global_best_reach": self.global_best_reach,
            "global_best_place": self.global_best_place,
            "avg_recent_reach_dist": np.mean(recent_reach) if recent_reach else float('inf'),
            "avg_recent_place_dist": np.mean(recent_place) if recent_place else float('inf'),
            "success_streak": self.curriculum.success_streak,
        }
    
    def format_progress(self, iteration: int, metrics: Dict[str, Any]) -> str:
        """Format progress for printing."""
        summary = self.get_progress_summary()
        
        phase_emoji = "🎯" if summary["current_phase"] == "reach" else "📦"
        
        lines = [
            f"\n{'='*80}",
            f"Iteration {iteration:5d} | Phase: {phase_emoji} {summary['current_phase'].upper()}",
            f"{'='*80}",
            f"  Episodes: {summary['total_episodes']:,}",
            f"  Reach Success: {summary['reach_success_rate']:6.2%} | "
            f"Place Success: {summary['place_success_rate']:6.2%}",
            f"  Best Reach Dist: {summary['global_best_reach']:.4f} m | "
            f"Best Place Dist: {summary['global_best_place']:.4f} m",
            f"  Avg Recent Reach: {summary['avg_recent_reach_dist']:.4f} m | "
            f"Avg Recent Place: {summary['avg_recent_place_dist']:.4f} m",
            f"  Success Streak: {summary['success_streak']}",
        ]
        
        if "policy_loss" in metrics:
            lines.append(
                f"  Policy Loss: {metrics['policy_loss']:.4f} | "
                f"Value Loss: {metrics['value_loss']:.4f} | "
                f"Entropy: {metrics['entropy']:.4f}"
            )
        
        lines.append(f"{'='*80}")
        
        return "\n".join(lines)

