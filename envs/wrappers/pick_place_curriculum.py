"""
Pick-and-Place with Curriculum Learning and Gradual Progress
=============================================================

This wrapper implements:
- 6D action space: [arm_flag, joint1, joint2, joint3, joint4, joint5]
- Joint 6 is always held at 0
- Gradual learning: memory of best control points, revert if worse
- Curriculum: reach phase first, then place phase
- Fixed object/target positions until success streak achieved
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.manipulation import AeroPiperPickPlace


@dataclass
class ProgressMemory:
    """Memory for tracking best control points and gradual progress."""
    
    # Best control values that achieved minimum distance
    best_ctrl: Optional[np.ndarray] = None
    best_dist: float = float('inf')
    
    # Previous step for comparison
    prev_ctrl: Optional[np.ndarray] = None
    prev_dist: float = float('inf')
    
    # Statistics
    improvements: int = 0
    regressions: int = 0
    reverts: int = 0
    total_steps: int = 0
    
    def reset(self, initial_ctrl: np.ndarray, initial_dist: float) -> None:
        """Reset at episode start."""
        self.best_ctrl = initial_ctrl.copy()
        self.best_dist = initial_dist
        self.prev_ctrl = initial_ctrl.copy()
        self.prev_dist = initial_dist
        self.improvements = 0
        self.regressions = 0
        self.reverts = 0
        self.total_steps = 0
    
    def update(
        self, 
        new_ctrl: np.ndarray, 
        new_dist: float,
        tolerance: float = 0.002,
    ) -> Tuple[bool, bool, Optional[np.ndarray]]:
        """
        Update memory with new control point.
        
        Args:
            new_ctrl: New control values
            new_dist: New distance to target
            tolerance: Distance tolerance for considering improvement
            
        Returns:
            (improved, should_revert, revert_ctrl)
        """
        self.total_steps += 1
        improved = False
        should_revert = False
        revert_ctrl = None
        
        # Check if this is a new best
        if new_dist < self.best_dist - tolerance:
            self.best_dist = new_dist
            self.best_ctrl = new_ctrl.copy()
            self.prev_dist = new_dist
            self.prev_ctrl = new_ctrl.copy()
            self.improvements += 1
            improved = True
        elif new_dist > self.prev_dist + tolerance:
            # Got worse - should revert to best known position
            self.regressions += 1
            should_revert = True
            revert_ctrl = self.best_ctrl
            self.reverts += 1
        else:
            # Roughly same - update prev for continued exploration
            self.prev_dist = new_dist
            self.prev_ctrl = new_ctrl.copy()
        
        return improved, should_revert, revert_ctrl
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "best_dist": self.best_dist,
            "prev_dist": self.prev_dist,
            "improvements": self.improvements,
            "regressions": self.regressions,
            "reverts": self.reverts,
            "total_steps": self.total_steps,
            "improvement_rate": self.improvements / max(1, self.total_steps),
        }


class CurriculumPhase(Enum):
    """Training phases for curriculum learning."""
    REACH = 0  # Learn to reach the cube
    PLACE = 1  # Learn to place at target (after reaching)


class AeroPiperPickPlaceCurriculum(gym.Env):
    """
    Pick-and-place environment with curriculum learning and gradual progress.
    
    Key Features:
    -------------
    1. 6D Action Space: [arm_select, j1, j2, j3, j4, j5]
       - arm_select >= 0: right arm, < 0: left arm
       - j1-j5: Control for joints 1-5
       - Joint 6 is always held at 0
    
    2. Gradual Learning:
       - Tracks best control points that reduced distance
       - Reverts to best position when action increases distance
       - Encourages step-by-step progress toward target
    
    3. Curriculum Learning:
       - Phase 1 (REACH): Focus on getting palm close to cube
       - Phase 2 (PLACE): Focus on getting cube close to target
       - Transition when reach success rate > threshold
    
    4. Fixed Positions Until Success:
       - Cube and target positions stay fixed
       - Only randomize after achieving success streak
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    # Class-level curriculum state (shared across all instances)
    # These track EPISODE-level successes, not step-level
    _global_reach_successes: List[float] = []
    _global_place_successes: List[float] = []
    _global_reach_best: float = float('inf')
    _global_place_best: float = float('inf')
    _global_success_streak: int = 0
    _global_phase: CurriculumPhase = CurriculumPhase.REACH
    _phase_transition_threshold: float = 0.90  # 90% reach success to unlock place
    _phase_window: int = 200  # Last N EPISODES to average for phase transition (more stable)
    _randomize_streak_threshold: int = 10  # Consecutive episode successes before randomizing
    _total_episodes: int = 0
    _curriculum_initialized: bool = False
    _min_episodes_for_transition: int = 500  # Minimum episodes before phase transition (higher for stability)
    _training_started: bool = False  # Flag to prevent counting resets before training
    
    def __init__(
        self,
        max_episode_steps: int = 2000,
        reward_type: str = "dense",
        success_threshold: float = 0.10,  # 10cm
        randomize_objects: bool = False,  # Start fixed, unlock via curriculum
        action_scale: float = 0.3,  # Smaller for more gradual movement
        frame_skip: int = 5,
        render_mode: Optional[str] = None,
        # Gradual learning parameters
        distance_tolerance: float = 0.002,  # 2mm tolerance
        revert_on_regression: bool = True,
        # Curriculum parameters
        phase_transition_threshold: float = 0.80,
        randomize_after_streak: int = 5,
        disable_curriculum: bool = False,  # If True, train reach+place together
    ) -> None:
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0
        self.success_threshold = success_threshold
        self._action_scale = action_scale
        self._distance_tolerance = distance_tolerance
        self._revert_on_regression = revert_on_regression
        self._base_randomize = randomize_objects
        
        # Update class-level parameters
        AeroPiperPickPlaceCurriculum._phase_transition_threshold = phase_transition_threshold
        AeroPiperPickPlaceCurriculum._randomize_streak_threshold = randomize_after_streak
        self._disable_curriculum = disable_curriculum
        
        if disable_curriculum:
            # Force PLACE phase (reward both reach and place)
            AeroPiperPickPlaceCurriculum._global_phase = CurriculumPhase.PLACE
        
        # Create underlying environment
        self._env = AeroPiperPickPlace(
            reward_type=reward_type,
            success_threshold=success_threshold,
            randomize_objects=randomize_objects,  # Pass through the flag
            action_scale=action_scale,
            frame_skip=frame_skip,
        )
        
        # 6D Action space: [arm_flag, j1, j2, j3, j4, j5]
        # Joint 6 is always 0
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        
        # Observation space: base obs + phase info + distances
        sample_obs = self._env.reset()
        # Base obs + [phase_flag, reach_dist, place_dist, arm_flag, 
        #             best_reach_dist, best_place_dist, dist_improvement]
        obs_dim = sample_obs.shape[0] + 7
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Store references
        self.model = self._env.model
        self.data = self._env.data
        
        # Actuator indices (26 total actuators)
        # Right arm: 0-5, Right fingers: 6-12, Left arm: 13-18, Left fingers: 19-25
        self._right_arm_idx = list(range(0, 6))
        self._right_finger_idx = list(range(6, 13))
        self._left_arm_idx = list(range(13, 19))
        self._left_finger_idx = list(range(19, 26))
        
        # Ctrl ranges
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        
        # Episode state
        self._active_arm = "right"
        self._home_ctrl: Optional[np.ndarray] = None
        
        # Progress memory for each phase
        self._reach_memory = ProgressMemory()
        self._place_memory = ProgressMemory()
        
        # Episode tracking
        self._episode_reach_success = False
        self._episode_place_success = False
        self._episode_best_reach = float('inf')
        self._episode_best_place = float('inf')
        
        # External curriculum control (from training loop)
        self.reach_only = False  # Can be set externally
    
    @classmethod
    def get_curriculum_phase(cls) -> CurriculumPhase:
        """Get current curriculum phase."""
        return cls._global_phase
    
    @classmethod
    def reset_curriculum(cls):
        """Reset curriculum state (call at start of training)."""
        cls._global_reach_successes = []
        cls._global_place_successes = []
        cls._global_reach_best = float('inf')
        cls._global_place_best = float('inf')
        cls._global_success_streak = 0
        cls._global_phase = CurriculumPhase.REACH
        cls._total_episodes = 0
        cls._curriculum_initialized = True
        cls._training_started = False  # Will be set to True after first rollout
    
    @classmethod
    def start_training(cls):
        """Mark that training has started (call after env creation)."""
        cls._training_started = True
    
    @classmethod
    def update_global_curriculum(
        cls, 
        reach_success: bool, 
        place_success: bool,
        reach_dist: float,
        place_dist: float,
    ) -> Dict[str, Any]:
        """Update global curriculum state at episode end."""
        # Initialize if needed
        if not cls._curriculum_initialized:
            cls.reset_curriculum()
        
        # Only count episodes after training has actually started
        if not cls._training_started:
            return {
                "phase": cls._global_phase.name,
                "phase_changed": False,
                "reach_rate": 0.0,
                "place_rate": 0.0,
                "global_reach_best": cls._global_reach_best,
                "global_place_best": cls._global_place_best,
                "success_streak": 0,
                "total_episodes": 0,
                "should_randomize": False,
            }
        
        cls._total_episodes += 1
        cls._global_reach_successes.append(float(reach_success))
        cls._global_place_successes.append(float(place_success))
        
        # Update best distances
        if reach_dist < cls._global_reach_best:
            cls._global_reach_best = reach_dist
        if place_dist < cls._global_place_best:
            cls._global_place_best = place_dist
        
        # Update success streak (based on phase-appropriate success)
        if cls._global_phase == CurriculumPhase.REACH:
            if reach_success:
                cls._global_success_streak += 1
            else:
                cls._global_success_streak = 0
        else:
            if reach_success and place_success:
                cls._global_success_streak += 1
            else:
                cls._global_success_streak = 0
        
        # Calculate recent success rates (from episode history)
        recent_reach = cls._global_reach_successes[-cls._phase_window:]
        recent_place = cls._global_place_successes[-cls._phase_window:]
        reach_rate = np.mean(recent_reach) if len(recent_reach) >= 10 else 0.0
        place_rate = np.mean(recent_place) if len(recent_place) >= 10 else 0.0
        
        # Phase transition - require minimum episodes before transitioning
        old_phase = cls._global_phase
        if cls._global_phase == CurriculumPhase.REACH:
            # Need at least _min_episodes_for_transition episodes and 90% success rate to transition
            if cls._total_episodes >= cls._min_episodes_for_transition and reach_rate >= cls._phase_transition_threshold:
                cls._global_phase = CurriculumPhase.PLACE
                print(f"\n{'*'*60}")
                print(f"🎉 CURRICULUM: Transitioning to PLACE phase!")
                print(f"   Episodes: {cls._total_episodes}")
                print(f"   Reach success rate: {reach_rate:.2%}")
                print(f"   Best reach distance: {cls._global_reach_best:.4f}m")
                print(f"{'*'*60}\n")
        
        return {
            "phase": cls._global_phase.name,
            "phase_changed": old_phase != cls._global_phase,
            "reach_rate": reach_rate,
            "place_rate": place_rate,
            "global_reach_best": cls._global_reach_best,
            "global_place_best": cls._global_place_best,
            "success_streak": cls._global_success_streak,
            "total_episodes": cls._total_episodes,
            "should_randomize": cls._global_success_streak >= cls._randomize_streak_threshold,
        }
    
    def _get_positions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get cube, target, right EE, and left EE positions."""
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id].copy()
        target_pos = self._env.data.site_xpos[self._env.target_site_id].copy()
        right_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["right"])
        ].copy()
        left_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["left"])
        ].copy()
        return cube_pos, target_pos, right_ee, left_ee
    
    def _get_active_ee(self) -> np.ndarray:
        """Get position of active arm end-effector."""
        site_name = self._env.ee_sites[self._active_arm]
        site_id = self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self._env.data.site_xpos[site_id].copy()
    
    def _distance_to_cube(self) -> float:
        """Distance from active EE to cube."""
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id]
        ee_pos = self._get_active_ee()
        return float(np.linalg.norm(cube_pos - ee_pos))
    
    def _distance_cube_to_target(self) -> float:
        """Distance from cube to target (for reference)."""
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id]
        target_pos = self._env.data.site_xpos[self._env.target_site_id]
        return float(np.linalg.norm(cube_pos - target_pos))
    
    def _distance_palm_to_target(self) -> float:
        """Distance from active palm to target (for PLACE phase)."""
        ee_pos = self._get_active_ee()
        target_pos = self._env.data.site_xpos[self._env.target_site_id]
        return float(np.linalg.norm(ee_pos - target_pos))
    
    def _expand_action(self, action: np.ndarray) -> np.ndarray:
        """
        Expand 6D action to 26D full action.
        
        Input: [arm_flag, j1, j2, j3, j4, j5]
        Output: 26D action for all actuators
        """
        # Determine active arm from first action component
        self._active_arm = "right" if action[0] >= 0 else "left"
        
        # Build 6D arm action (5 joints + joint6=0)
        arm_joints = np.zeros(6, dtype=np.float64)
        arm_joints[:5] = action[1:6]  # Joints 1-5 from policy
        arm_joints[5] = 0.0  # Joint 6 always 0
        
        # Get current control values
        current_ctrl = self._env.ctrl_cache.copy()
        full_action = np.zeros(26, dtype=np.float64)
        
        if self._active_arm == "right":
            active_arm_idx = self._right_arm_idx
            active_finger_idx = self._right_finger_idx
            inactive_arm_idx = self._left_arm_idx
            inactive_finger_idx = self._left_finger_idx
        else:
            active_arm_idx = self._left_arm_idx
            active_finger_idx = self._left_finger_idx
            inactive_arm_idx = self._right_arm_idx
            inactive_finger_idx = self._right_finger_idx
        
        # Apply arm action to active arm
        full_action[active_arm_idx] = arm_joints
        
        # Keep fingers open (no grasping for now)
        full_action[active_finger_idx] = 0.0
        
        # Return inactive arm to home position (if we have home)
        if self._home_ctrl is not None:
            home_arm = self._home_ctrl[inactive_arm_idx]
            current_arm = current_ctrl[inactive_arm_idx]
            arm_delta = (home_arm - current_arm) / self._action_scale
            arm_delta = np.clip(arm_delta, -1.0, 1.0)
            full_action[inactive_arm_idx] = arm_delta
            
            # Keep inactive fingers at home
            home_fingers = self._home_ctrl[inactive_finger_idx]
            current_fingers = current_ctrl[inactive_finger_idx]
            finger_delta = (home_fingers - current_fingers) / self._action_scale
            finger_delta = np.clip(finger_delta, -1.0, 1.0)
            full_action[inactive_finger_idx] = finger_delta
        
        return full_action
    
    def _get_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Build observation with curriculum and progress info."""
        phase_flag = 0.0 if self._global_phase == CurriculumPhase.REACH else 1.0
        reach_dist = self._distance_to_cube()
        place_dist = self._distance_palm_to_target()  # Palm to target for place phase
        arm_flag = 1.0 if self._active_arm == "right" else -1.0
        
        # Progress info
        best_reach = min(self._reach_memory.best_dist, reach_dist)
        best_place = min(self._place_memory.best_dist, place_dist)
        
        # Distance improvement from episode start (for current phase)
        if self._global_phase == CurriculumPhase.REACH or not self._episode_reach_success:
            dist_improvement = (self._episode_best_reach - reach_dist) if self._episode_best_reach < float('inf') else 0.0
        else:
            dist_improvement = (self._episode_best_place - place_dist) if self._episode_best_place < float('inf') else 0.0
        
        return np.concatenate([
            base_obs,
            [phase_flag, reach_dist, place_dist, arm_flag,
             best_reach, best_place, dist_improvement]
        ]).astype(np.float32)
    
    def _compute_reward(
        self,
        base_reward: float,
        reach_dist: float,
        place_dist: float,  # This is now palm-to-target distance
        reach_improved: bool,
        place_improved: bool,
        reach_reverted: bool,
        place_reverted: bool,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute shaped reward enforcing SEQUENTIAL behavior.
        
        CRITICAL: Within each episode:
        - Phase 1: ONLY reward reaching the cube (ignore target completely)
        - Phase 2: ONLY after reach achieved, reward going to target
        
        This ensures the robot MUST reach first, then place.
        """
        components = {}
        
        # EPISODE-LEVEL phase (not global curriculum) determines reward
        # This is the key change - we use episode state, not global phase
        episode_in_reach_phase = not self._episode_reach_success
        
        if episode_in_reach_phase:
            # ============================================
            # PHASE 1: REACH THE CUBE (ignore target!)
            # ============================================
            prev_dist = self._reach_memory.prev_dist
            
            # Distance-based reward (closer to cube = better)
            components["reach_distance"] = -reach_dist * 5.0
            
            # Improvement bonus (getting closer to cube)
            dist_change = prev_dist - reach_dist
            components["reach_progress"] = dist_change * 30.0
            
            # Big bonus when reaching the cube (unlocks place phase!)
            if reach_dist < self.success_threshold:
                components["reach_success"] = 50.0  # Increased to encourage reaching first
            
            # Improvement flag bonus
            if reach_improved:
                components["reach_improvement"] = 1.0
            
            # Penalty for regression
            if reach_reverted:
                components["reach_regression"] = -0.5
            
            # IMPORTANT: NO reward for target distance in this phase!
            # This forces robot to focus on reaching first
            components["place_distance"] = 0.0
            components["place_progress"] = 0.0
                
        else:
            # ============================================
            # PHASE 2: GO TO TARGET (after reaching cube)
            # ============================================
            prev_dist = self._place_memory.prev_dist
            
            # Constant bonus for being in place phase (reached cube!)
            components["reached_bonus"] = 2.0
            
            # Distance to target (palm to target) - STRONG reward
            components["place_distance"] = -place_dist * 20.0  # Increased from 5
            
            # Progress toward target - VERY STRONG
            dist_change = prev_dist - place_dist
            components["place_progress"] = dist_change * 100.0  # Increased from 50
            
            # MASSIVE bonus when reaching target (completing the sequence!)
            if place_dist < self.success_threshold:
                components["place_success"] = 500.0  # HUGE bonus for sequential completion!
            
            # Improvement flag bonus
            if place_improved:
                components["place_improvement"] = 5.0
            
            # No penalty for regression in place phase - encourage exploration
            components["place_regression"] = 0.0
            
            # No reach reward in this phase (already achieved)
            components["reach_distance"] = 0.0
            components["reach_progress"] = 0.0
        
        # Small base reward component
        components["base"] = base_reward * 0.05
        
        total = sum(components.values())
        return total, components
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()
        
        # Randomization: if _base_randomize is True, always randomize
        # (The old curriculum logic required success streak, but that's confusing)
        self._env.randomize_objects = self._base_randomize
        
        # Reset base environment
        base_obs = self._env.reset(rng=self._np_random)
        self._step_count = 0
        
        # Store home position
        self._home_ctrl = self._env.ctrl_cache.copy()
        
        # Initialize distances
        reach_dist = self._distance_to_cube()
        place_dist = self._distance_palm_to_target()  # Palm to target for place phase
        
        # Reset progress memory
        self._reach_memory.reset(self._env.ctrl_cache.copy(), reach_dist)
        self._place_memory.reset(self._env.ctrl_cache.copy(), place_dist)
        
        # Reset episode tracking
        self._episode_reach_success = False
        self._episode_place_success = False
        self._episode_best_reach = reach_dist
        self._episode_best_place = place_dist
        
        # Update reach_only from curriculum
        self.reach_only = self._global_phase == CurriculumPhase.REACH
        self._env.reach_only = self.reach_only
        
        obs = self._get_obs(base_obs)
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step."""
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        
        # Update active arm from action
        self._active_arm = "right" if action[0] >= 0 else "left"
        
        # Get current distances and ctrl before action
        pre_reach_dist = self._distance_to_cube()
        pre_place_dist = self._distance_palm_to_target()  # Palm to target for place phase
        pre_ctrl = self._env.ctrl_cache.copy()
        
        # Expand and execute action
        full_action = self._expand_action(action)
        base_obs, base_reward, done, env_info = self._env.step(full_action)
        self._step_count += 1
        
        # Get new distances and ctrl after action
        post_reach_dist = self._distance_to_cube()
        post_place_dist = self._distance_palm_to_target()  # Palm to target for place phase
        post_ctrl = self._env.ctrl_cache.copy()
        
        # Update progress memory based on current phase
        # In REACH phase: track reach distance
        # In PLACE phase (after reaching): track place distance
        if not self._episode_reach_success:
            # Still in reach phase - track reach progress
            reach_improved, reach_should_revert, reach_revert_ctrl = self._reach_memory.update(
                post_ctrl, post_reach_dist, self._distance_tolerance
            )
            place_improved = False
            place_should_revert = False
            place_revert_ctrl = None
        else:
            # Reach achieved - now track place progress
            reach_improved = False
            reach_should_revert = False
            reach_revert_ctrl = None
            place_improved, place_should_revert, place_revert_ctrl = self._place_memory.update(
                post_ctrl, post_place_dist, self._distance_tolerance
            )
        
        # Determine phase based on EPISODE state (not global curriculum)
        # This ensures sequential behavior: first reach, then place
        episode_in_reach_phase = not self._episode_reach_success
        
        reach_reverted = False
        place_reverted = False
        
        if self._revert_on_regression:
            if episode_in_reach_phase and reach_should_revert and reach_revert_ctrl is not None:
                # In reach phase - revert to best reach position
                self._env.ctrl_cache = reach_revert_ctrl.copy()
                self._env.data.ctrl[self._env.actuator_ids] = reach_revert_ctrl
                reach_reverted = True
            elif not episode_in_reach_phase and place_should_revert and place_revert_ctrl is not None:
                # In place phase - revert to best place position
                self._env.ctrl_cache = place_revert_ctrl.copy()
                self._env.data.ctrl[self._env.actuator_ids] = place_revert_ctrl
                place_reverted = True
        
        # Update episode tracking
        was_reach_success = self._episode_reach_success
        if post_reach_dist < self.success_threshold:
            self._episode_reach_success = True
        
        # Track best reach distance
        if post_reach_dist < self._episode_best_reach:
            self._episode_best_reach = post_reach_dist
        
        # Place success and tracking only counts AFTER reach is achieved
        if self._episode_reach_success:
            if post_place_dist < self.success_threshold:
                self._episode_place_success = True
            # Only update best place distance after reach is achieved
            if post_place_dist < self._episode_best_place:
                self._episode_best_place = post_place_dist
        
        # Reset place memory when transitioning from reach to place
        if self._episode_reach_success and not was_reach_success:
            # Just achieved reach - reset place memory from current position
            self._place_memory.reset(post_ctrl, post_place_dist)
        
        # Compute shaped reward
        reward, reward_components = self._compute_reward(
            base_reward,
            post_reach_dist,
            post_place_dist,
            reach_improved,
            place_improved,
            reach_reverted,
            place_reverted,
        )
        
        # Build observation
        obs = self._get_obs(base_obs)
        
        # Termination
        truncated = self._step_count >= self.max_episode_steps
        # Early termination on full success (reach + place achieved)
        full_success = self._episode_reach_success and self._episode_place_success
        terminated = (done or full_success) and not truncated
        
        # Build info
        info = self._get_info()
        # Save our calculated success values before updating with base env info
        our_success_reach = info["success_reach"]
        our_success_place = info["success_place"]
        our_success = info["success"]
        # Update with base env info (but it will overwrite success_place incorrectly)
        info.update(env_info)
        # Restore OUR success calculations (palm-to-target based, not cube-to-target)
        info["success_reach"] = our_success_reach
        info["success_place"] = our_success_place
        info["success"] = our_success
        info["reach_improved"] = reach_improved
        info["place_improved"] = place_improved
        info["reach_reverted"] = reach_reverted
        info["place_reverted"] = place_reverted
        info["reward_components"] = reward_components
        info.update(self._reach_memory.get_stats())
        
        # Update global curriculum on episode end
        if terminated or truncated:
            # For curriculum purposes, use the BEST distance achieved in the episode
            # This is more meaningful than momentary success
            episode_reach_success = self._episode_best_reach < self.success_threshold
            episode_place_success = self._episode_best_place < self.success_threshold
            
            curriculum_info = self.update_global_curriculum(
                episode_reach_success,
                episode_place_success,
                self._episode_best_reach,
                self._episode_best_place,
            )
            info.update(curriculum_info)
            info["episode_reach_success"] = episode_reach_success
            info["episode_place_success"] = episode_place_success
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        """Get current info dict."""
        cube_pos, target_pos, right_ee, left_ee = self._get_positions()
        ee_pos = self._get_active_ee()
        
        # Phase 1 (REACH): palm to cube distance
        reach_dist = np.linalg.norm(cube_pos - ee_pos)
        
        # Phase 2 (PLACE): palm to target distance (not cube to target!)
        # Since we're not grasping, "placing" means moving palm to target after reaching cube
        place_dist = np.linalg.norm(ee_pos - target_pos)
        
        # Also track cube to target for reference
        cube_to_target = np.linalg.norm(cube_pos - target_pos)
        
        # Step-level reach success (is palm near cube RIGHT NOW?)
        step_reach_success = reach_dist < self.success_threshold
        
        # Episode-level reach success (has reach been achieved at any point in this episode?)
        # This is what matters for the sequential task
        episode_has_reached = self._episode_reach_success
        
        # Place success: palm within 10cm of target AFTER having reached the cube
        # This requires episode_reach_success to be True first
        place_success = episode_has_reached and place_dist < self.success_threshold
        
        return {
            "step_count": self._step_count,
            "max_episode_steps": self.max_episode_steps,
            "cube_pos": cube_pos.copy(),
            "target_pos": target_pos.copy(),
            "ee_pos": ee_pos.copy(),
            "dist_cube_hand": float(reach_dist),
            "dist_palm_target": float(place_dist),  # Palm to target (for PLACE phase)
            "dist_cube_target": float(cube_to_target),  # Cube to target (reference)
            "dist_cube_right_hand": float(np.linalg.norm(cube_pos - right_ee)),
            "dist_cube_left_hand": float(np.linalg.norm(cube_pos - left_ee)),
            # Step-level: is palm near cube right now?
            "success_reach_step": step_reach_success,
            # Episode-level: has reach been achieved in this episode? (used for sequential tracking)
            "success_reach": episode_has_reached,  # This is what matters for sequential success
            "success_place": place_success,
            "success": episode_has_reached and place_success and not self.reach_only,
            "active_arm": self._active_arm,
            "curriculum_phase": self._global_phase.name,
            "global_reach_best": self._global_reach_best,
            "global_place_best": self._global_place_best,
            "success_streak": self._global_success_streak,
            "episode_reach_success": self._episode_reach_success,
            "episode_place_success": self._episode_place_success,
            "episode_best_reach": self._episode_best_reach,
            "episode_best_place": self._episode_best_place,
            "episode_phase": "PLACE" if self._episode_reach_success else "REACH",
        }
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode in ["rgb_array", "human"]:
            return self._env.render(width=640, height=480)
        return None
    
    def close(self) -> None:
        """Close the environment."""
        pass
    
    @property
    def np_random(self) -> np.random.Generator:
        """Get random generator."""
        if not hasattr(self, "_np_random"):
            self._np_random = np.random.default_rng()
        return self._np_random

