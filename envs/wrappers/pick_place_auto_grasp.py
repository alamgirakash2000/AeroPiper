"""
Pick-and-Place with Guided Arm Selection
=========================================

- Policy chooses the active arm each step (no auto-arm chooser by default).
- Policy controls only the first 5 joints of the chosen arm; joint6 is held.
- Grasping can be disabled (default) to focus on reach/place progress.
- Built-in guard rails revert to the last pose that improved progress.
- Curriculum: keep cube/target fixed until a streak of successes is achieved.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple, Set

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.manipulation import AeroPiperPickPlace


class GraspState(Enum):
    """Grasp state machine states."""
    REACHING = 0      # Hand open, moving toward object
    GRASPING = 1      # Near object, closing fingers
    HOLDING = 2       # Object grasped, maintaining grip
    RELEASING = 3     # At target, opening fingers
    FAILED_GRASP = 4  # Failed to grasp (grip maxed, nothing grabbed)


class AeroPiperPickPlaceAutoGrasp(gym.Env):
    """
    Pick-and-place with automatic grasp management and policy-driven arm choice.
    
    Grasp Detection Methods:
    1. Contact detection - fingers touching cube
    2. Height detection - cube lifted above table
    3. Velocity correlation - cube moves with hand
    
    Policy only controls: [arm_flag, j1..j5]; joint6 held, grip automatic.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        max_episode_steps: int = 2000,
        reward_type: str = "dense",
        success_threshold: float = 0.10,
        randomize_objects: bool = False,
        action_scale: float = 0.5,
        frame_skip: int = 5,
        render_mode: Optional[str] = None,
        auto_select_arm: bool = False,
        enable_grasping: bool = False,
        curriculum_successes: int = 5,
        progress_tolerance: float = 0.0,
        # Grasp parameters
        approach_distance: float = 0.10,      # Start closing at 10cm
        grasp_trigger_distance: float = 0.06, # Definitely close at 6cm
        grasp_close_speed: float = 0.03,      # Grip increment per step
        max_grip_level: float = 0.90,         # Max grip before "failed"
        release_trigger_distance: float = 0.08,  # Start releasing near target
        table_height: float = 0.67,           # Table surface z
        lift_threshold: float = 0.03,         # 3cm above table = lifted
    ) -> None:
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0
        self.success_threshold = success_threshold
        self._action_scale = action_scale
        self._auto_select_arm = auto_select_arm
        self._grasp_enabled = enable_grasping
        self._progress_tolerance = progress_tolerance
        self._base_randomize = randomize_objects
        self._curriculum_threshold = max(1, curriculum_successes)
        self._success_streak = 0
        
        # Grasp parameters
        self.approach_distance = approach_distance
        self.grasp_trigger_distance = grasp_trigger_distance
        self.grasp_close_speed = grasp_close_speed
        self.max_grip_level = max_grip_level
        self.release_trigger_distance = release_trigger_distance
        self.table_height = table_height
        self.lift_threshold = lift_threshold
        
        # Create underlying environment
        self._env = AeroPiperPickPlace(
            reward_type=reward_type,
            success_threshold=success_threshold,
            randomize_objects=randomize_objects,
            action_scale=action_scale,
            frame_skip=frame_skip,
        )
        
        # Action space: 6D (arm flag + 5 joints, joint6 held at 0)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),
            dtype=np.float32,
        )
        
        # Observation space
        sample_obs = self._env.reset()
        # Original obs + grasp_state(1) + grip_level(1) + is_grasped(1) +
        # cube_lifted(1) + dist_right(1) + dist_left(1) + active_arm_flag(1) +
        # phase_flag(1: 0=reaching, 1=placing)
        obs_dim = sample_obs.shape[0] + 8
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # Store references
        self.model = self._env.model
        self.data = self._env.data
        
        # Actuator indices
        self._right_arm_idx = list(range(0, 6))
        self._right_finger_idx = list(range(6, 13))
        self._left_arm_idx = list(range(13, 19))
        self._left_finger_idx = list(range(19, 26))
        
        # Ctrl ranges
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        
        # Get cube body ID for contact detection
        self._cube_body_id = self._env.cube_body_id
        
        # Get finger body IDs for contact detection
        self._right_finger_bodies = self._get_finger_body_ids("right")
        self._left_finger_bodies = self._get_finger_body_ids("left")
        
        # Grasp state
        self._grasp_state = GraspState.REACHING
        self._grip_level = 0.0  # 0 = open, 1 = closed
        self._object_grasped = False
        self._grasp_confirmed_steps = 0  # Steps with grasp confirmed
        self._home_ctrl = None
        self._active_arm = "right"
        self._last_dist_right = 0.0
        self._last_dist_left = 0.0
        self._grasp_initiated = False
        self._initial_cube_pos = None
        self.reach_only = False
        
        # Failed grasp tracking
        self._consecutive_failed_grasps = 0
        self._max_failed_grasps = 3  # After this many failures, bigger penalty
        self._steps_since_last_grasp_attempt = 0
        self._min_steps_between_attempts = 30  # Force repositioning time
        
        # History for velocity tracking
        self._prev_cube_pos = None
        self._prev_ee_pos = None
        self._cube_lifted = False
        # Progress caches for revert-to-best behavior
        self._best_reach_dist = np.inf
        self._best_place_dist = np.inf
        self._best_reach_ctrl = None
        self._best_place_ctrl = None
        
    def _get_finger_body_ids(self, arm: str) -> Set[int]:
        """Get body IDs for finger collision detection."""
        finger_prefixes = [
            "index_", "middle_", "ring_", "pinky_", "thumb_",
            "palm", "if_", "mf_", "rf_", "pf_", "th_"
        ]
        if arm == "left":
            finger_prefixes = ["left_" + p for p in finger_prefixes] + ["left_palm"]
        
        body_ids = set()
        for idx in range(self.model.nbody):
            name = self.model.body(idx).name
            for prefix in finger_prefixes:
                if prefix in name:
                    body_ids.add(idx)
                    break
        return body_ids
    
    def _get_positions(self):
        """Get cube, target, and end-effector positions."""
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id].copy()
        target_pos = self._env.data.site_xpos[self._env.target_site_id].copy()
        right_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["right"])
        ].copy()
        left_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["left"])
        ].copy()
        
        if self._active_arm == "right":
            ee_pos = right_ee
            finger_bodies = self._right_finger_bodies
        else:
            ee_pos = left_ee
            finger_bodies = self._left_finger_bodies
        
        return cube_pos, target_pos, ee_pos, finger_bodies, right_ee, left_ee

    def _distance_to_cube(self, arm: str) -> float:
        """Distance from a specific arm end-effector to the cube."""
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id]
        ee_pos = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites[arm])
        ]
        return float(np.linalg.norm(cube_pos - ee_pos))

    def _distance_cube_to_target(self) -> float:
        """Distance from cube to target site."""
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id]
        target_pos = self._env.data.site_xpos[self._env.target_site_id]
        return float(np.linalg.norm(cube_pos - target_pos))

    def _select_active_arm(self):
        """Select the arm closest to the cube once per episode (pre-policy)."""
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id].copy()
        target_pos = self._env.data.site_xpos[self._env.target_site_id].copy()
        right_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["right"])
        ].copy()
        left_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["left"])
        ].copy()
        
        dist_right = np.linalg.norm(cube_pos - right_ee)
        dist_left = np.linalg.norm(cube_pos - left_ee)
        
        self._active_arm = "right" if dist_right <= dist_left else "left"
        self._last_dist_right = dist_right
        self._last_dist_left = dist_left
        self._initial_cube_pos = cube_pos.copy()
        return self._active_arm

    def _maybe_reselect_arm(self):
        """Optionally reselect arm if the cube has moved significantly."""
        if self._object_grasped:
            return  # keep arm once grasped
        if self._initial_cube_pos is None:
            return
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id].copy()
        if np.linalg.norm(cube_pos - self._initial_cube_pos) > 0.03:  # target moved
            self._select_active_arm()
    
    def _current_phase_distance(self) -> float:
        """Distance metric for progress: reach (EE→cube) or place (cube→target)."""
        if self._grasp_enabled:
            if self._object_grasped or self._cube_lifted or self._grasp_state in (GraspState.HOLDING, GraspState.RELEASING):
                return self._distance_cube_to_target()
        if self._env.reach_success:
            return self._distance_cube_to_target()
        return self._distance_to_cube(self._active_arm)

    def _track_progress(self, phase: str, new_dist: float) -> None:
        """Cache the best ctrl for the given phase when progress is made."""
        if phase == "reach" and new_dist < self._best_reach_dist:
            self._best_reach_dist = new_dist
            self._best_reach_ctrl = self._env.ctrl_cache.copy()
        elif phase == "place" and new_dist < self._best_place_dist:
            self._best_place_dist = new_dist
            self._best_place_ctrl = self._env.ctrl_cache.copy()

    def _revert_if_worse(self, phase: str, prev_dist: float, new_dist: float) -> float:
        """
        If distance worsens, snap controls back to the last improving point.
        Returns the possibly-updated new_dist (clamped to prev if reverted).
        """
        if new_dist <= prev_dist + self._progress_tolerance:
            return new_dist
        if phase == "reach" and self._best_reach_ctrl is not None:
            self._env.ctrl_cache = self._best_reach_ctrl.copy()
            self._env.data.ctrl[self._env.actuator_ids] = self._best_reach_ctrl
            return min(prev_dist, self._best_reach_dist)
        if phase == "place" and self._best_place_ctrl is not None:
            self._env.ctrl_cache = self._best_place_ctrl.copy()
            self._env.data.ctrl[self._env.actuator_ids] = self._best_place_ctrl
            return min(prev_dist, self._best_place_dist)
        return new_dist

    def _check_finger_cube_contact(self, finger_bodies: Set[int]) -> bool:
        """Check if any finger is in contact with cube."""
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            
            # Check if one is cube and other is finger
            if body1 == self._cube_body_id and body2 in finger_bodies:
                return True
            if body2 == self._cube_body_id and body1 in finger_bodies:
                return True
        return False
    
    def _check_cube_lifted(self, cube_pos: np.ndarray) -> bool:
        """Check if cube is lifted above table."""
        return cube_pos[2] > (self.table_height + self.lift_threshold)
    
    def _check_velocity_correlation(self, cube_pos: np.ndarray, ee_pos: np.ndarray) -> bool:
        """Check if cube is moving with hand (velocity correlation)."""
        if self._prev_cube_pos is None or self._prev_ee_pos is None:
            return False
        
        cube_vel = cube_pos - self._prev_cube_pos
        ee_vel = ee_pos - self._prev_ee_pos
        
        # If both have similar velocity direction and hand is moving
        ee_speed = np.linalg.norm(ee_vel)
        if ee_speed < 0.001:  # Hand not moving much
            return False
        
        # Check velocity correlation
        cube_speed = np.linalg.norm(cube_vel)
        if cube_speed < 0.0005:  # Cube not moving
            return False
        
        # Dot product of normalized velocities
        correlation = np.dot(cube_vel, ee_vel) / (cube_speed * ee_speed + 1e-8)
        return correlation > 0.5  # Moving in similar direction
    
    def _detect_grasp(self, cube_pos: np.ndarray, ee_pos: np.ndarray, 
                      finger_bodies: Set[int]) -> Tuple[bool, dict]:
        """
        Sophisticated grasp detection using multiple signals.
        
        Returns:
            is_grasped: Whether object is considered grasped
            signals: Dictionary of individual detection signals
        """
        dist_hand_to_cube = np.linalg.norm(ee_pos - cube_pos)
        
        # Signal 1: Contact detection
        has_contact = self._check_finger_cube_contact(finger_bodies)
        
        # Signal 2: Cube lifted above table
        is_lifted = self._check_cube_lifted(cube_pos)
        
        # Signal 3: Velocity correlation (cube moves with hand)
        vel_correlated = self._check_velocity_correlation(cube_pos, ee_pos)
        
        # Signal 4: Distance check (cube very close to hand)
        is_close = dist_hand_to_cube < 0.05
        
        # Signal 5: Grip is partially closed
        grip_engaged = self._grip_level > 0.15
        
        signals = {
            "contact": has_contact,
            "lifted": is_lifted,
            "velocity_correlated": vel_correlated,
            "close": is_close,
            "grip_engaged": grip_engaged,
        }
        
        # Grasp is confirmed if:
        # - Cube is lifted (strongest signal), OR
        # - Contact + close + grip engaged, OR
        # - Velocity correlated + close + grip engaged
        is_grasped = (
            is_lifted or
            (has_contact and is_close and grip_engaged) or
            (vel_correlated and is_close and grip_engaged)
        )
        
        return is_grasped, signals
    
    def _update_grasp_state(self):
        """Update grasp state machine."""
        cube_pos, target_pos, ee_pos, finger_bodies, _, _ = self._get_positions()
        
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        
        # Detect grasp status
        is_grasped, signals = self._detect_grasp(cube_pos, ee_pos, finger_bodies)
        self._cube_lifted = signals["lifted"]
        
        # Track steps since last grasp attempt
        self._steps_since_last_grasp_attempt += 1
        
        # State machine transitions
        if self._grasp_state == GraspState.REACHING:
            # Hand open, waiting to approach
            self._grip_level = max(0.0, self._grip_level - self.grasp_close_speed * 2)
            
            # Only attempt grasp if enough time has passed since last attempt
            can_attempt_grasp = self._steps_since_last_grasp_attempt >= self._min_steps_between_attempts
            
            if dist_to_cube < self.grasp_trigger_distance and can_attempt_grasp:
                # Close enough, start grasping
                self._grasp_state = GraspState.GRASPING
                self._grasp_confirmed_steps = 0
                self._steps_since_last_grasp_attempt = 0
            elif dist_to_cube < self.approach_distance:
                # Approaching, start pre-closing slightly
                self._grip_level = min(0.1, self._grip_level + self.grasp_close_speed * 0.5)
                
        elif self._grasp_state == GraspState.GRASPING:
            # Closing fingers around object
            self._grip_level += self.grasp_close_speed
            
            if is_grasped:
                # Grasp detected! Confirm for a few steps
                self._grasp_confirmed_steps += 1
                if self._grasp_confirmed_steps >= 3:
                    self._object_grasped = True
                    self._grasp_state = GraspState.HOLDING
                    # Reset failed grasp counter on success
                    self._consecutive_failed_grasps = 0
            else:
                self._grasp_confirmed_steps = 0
            
            # Check for failed grasp (grip maxed out, nothing grabbed)
            if self._grip_level >= self.max_grip_level and not is_grasped:
                self._grasp_state = GraspState.FAILED_GRASP
                self._consecutive_failed_grasps += 1
            
            # If hand moved away, go back to reaching
            if dist_to_cube > self.approach_distance:
                self._grasp_state = GraspState.REACHING
                self._grip_level = 0.0
                
        elif self._grasp_state == GraspState.HOLDING:
            # Object grasped, maintain grip
            # Slightly increase grip for security
            self._grip_level = min(self._grip_level + 0.005, self.max_grip_level)
            
            # Check if we should release (near target)
            if dist_to_target < self.release_trigger_distance and self._cube_lifted:
                self._grasp_state = GraspState.RELEASING
            
            # Check if grasp was lost
            if not is_grasped and not self._cube_lifted:
                self._grasp_confirmed_steps -= 1
                if self._grasp_confirmed_steps < -5:  # Lost for 5 steps
                    self._object_grasped = False
                    self._grasp_state = GraspState.REACHING
                    self._grip_level = 0.0
                    
        elif self._grasp_state == GraspState.RELEASING:
            # Near target, opening fingers
            self._grip_level -= self.grasp_close_speed * 2
            
            if self._grip_level <= 0.05:
                self._grip_level = 0.0
                self._object_grasped = False
                self._grasp_state = GraspState.REACHING
                
        elif self._grasp_state == GraspState.FAILED_GRASP:
            # Failed to grasp, open hand and reset
            self._grip_level -= self.grasp_close_speed * 3
            
            if self._grip_level <= 0.05:
                self._grip_level = 0.0
                self._grasp_state = GraspState.REACHING
                # Force policy to reposition by waiting before next attempt
                self._steps_since_last_grasp_attempt = 0
        
        # Clamp grip level
        self._grip_level = np.clip(self._grip_level, 0.0, 1.0)
        
        # Update history
        self._prev_cube_pos = cube_pos.copy()
        self._prev_ee_pos = ee_pos.copy()
    
    def _get_finger_ctrl_targets(self, grip_level: float, finger_idx: list) -> np.ndarray:
        """Convert grip level [0, 1] to finger ctrl targets."""
        targets = np.zeros(len(finger_idx))
        
        for i, act_idx in enumerate(finger_idx):
            low = self._ctrl_range[act_idx, 0]
            high = self._ctrl_range[act_idx, 1]
            
            trn_type = self.model.actuator_trntype[act_idx]
            
            if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                # Thumb abduction: higher = more adducted
                targets[i] = low + grip_level * (high - low)
            else:
                # Tendons: lower = more flexed
                targets[i] = high - grip_level * (high - low)
        
        return targets
    
    def _expand_action(self, arm_action: np.ndarray) -> np.ndarray:
        """Expand 6D (arm flag + 5 joints) action to 26D full action with automatic grip."""
        # arm_action[0] chooses arm, [1:6] are joint deltas, joint6 is frozen at 0.
        arm_selector = arm_action[0]
        self._active_arm = "right" if arm_selector >= 0 else "left"
        arm_joints = np.zeros(6, dtype=np.float64)
        arm_joints[:5] = arm_action[1:6]
        arm_joints[5] = 0.0  # hold 6th joint steady
        
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
        
        # Active arm: apply joint actions
        if not self._grasp_enabled:
            full_action[active_arm_idx] = arm_joints
        else:
            # Grasp-enabled: optionally freeze during grasp phases
            if self._grasp_state == GraspState.GRASPING:
                full_action[active_arm_idx] = 0.0  # Hold current position
            elif self._grasp_state == GraspState.RELEASING:
                full_action[active_arm_idx] = 0.0
            else:
                full_action[active_arm_idx] = arm_joints
        
        # Active fingers: controlled by grasp state machine (or held open)
        if self._grasp_enabled:
            finger_targets = self._get_finger_ctrl_targets(self._grip_level, active_finger_idx)
            finger_current = current_ctrl[active_finger_idx]
            finger_delta = (finger_targets - finger_current) / self._action_scale
            finger_delta = np.clip(finger_delta, -1.0, 1.0)
            full_action[active_finger_idx] = finger_delta
        else:
            full_action[active_finger_idx] = 0.0  # keep fingers steady/open
        
        # Inactive arm: return to home
        if self._home_ctrl is not None:
            home_arm = self._home_ctrl[inactive_arm_idx]
            current_arm = current_ctrl[inactive_arm_idx]
            arm_delta = (home_arm - current_arm) / self._action_scale
            arm_delta = np.clip(arm_delta, -1.0, 1.0)
            full_action[inactive_arm_idx] = arm_delta
            
            home_fingers = self._home_ctrl[inactive_finger_idx]
            current_fingers = current_ctrl[inactive_finger_idx]
            finger_home_delta = (home_fingers - current_fingers) / self._action_scale
            finger_home_delta = np.clip(finger_home_delta, -1.0, 1.0)
            full_action[inactive_finger_idx] = finger_home_delta
        
        return full_action
    
    def _get_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Augment observation with grasp state info."""
        grasp_state_norm = self._grasp_state.value / 4.0  # Normalize
        is_grasped = 1.0 if self._object_grasped else 0.0
        is_lifted = 1.0 if self._cube_lifted else 0.0
        phase_flag = 1.0 if (self._env.reach_success or self._object_grasped or self._cube_lifted or self._grasp_state in (GraspState.GRASPING, GraspState.HOLDING, GraspState.RELEASING)) else 0.0
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id]
        right_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["right"])
        ]
        left_ee = self._env.data.site_xpos[
            self._env._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self._env.ee_sites["left"])
        ]
        dist_right = np.linalg.norm(cube_pos - right_ee)
        dist_left = np.linalg.norm(cube_pos - left_ee)
        active_arm_flag = 1.0 if self._active_arm == "right" else -1.0
        
        return np.concatenate([
            base_obs,
            [
                grasp_state_norm,
                self._grip_level,
                is_grasped,
                is_lifted,
                dist_right,
                dist_left,
                active_arm_flag,
                phase_flag,
            ]
        ]).astype(np.float32)
    
    def _shape_reward(self, base_reward: float, prev_dist: float, new_dist: float, phase: str) -> float:
        """Add reward shaping based on stepwise progress (and optional grasp)."""
        reward = base_reward
        # Direct progress encouragement
        reward += (prev_dist - new_dist)
        dist_to_target = self._distance_cube_to_target()
        # Only shape toward target once reach is achieved or we're in place phase.
        if (self._env.reach_success and not self.reach_only) or self._object_grasped or phase == "place":
            reward -= 0.1 * dist_to_target

        if not self._grasp_enabled:
            # Curriculum-style bonuses for hitting the two distance thresholds.
            if phase == "reach" and new_dist <= 0.10:
                reward += 0.5
            if phase == "place" and dist_to_target <= 0.10:
                reward += 1.0
            return reward

        # Grasp-enabled bonuses
        if self._grasp_state == GraspState.GRASPING and not self._grasp_initiated:
            reward += 0.5
            self._grasp_initiated = True
        if self._cube_lifted and self._object_grasped:
            reward += 1.0
        if self._grasp_state == GraspState.RELEASING:
            reward += 0.3
        if dist_to_target <= 0.10:
            reward += 1.0
        return reward
    
    def _capture_home_ctrl(self):
        """Capture home position ctrl values."""
        self._home_ctrl = self._env.ctrl_cache.copy()
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        else:
            self._np_random = np.random.default_rng()
        
        # Curriculum: hold object/target fixed until a streak of successes is seen.
        curriculum_randomize = self._base_randomize and (self._success_streak >= self._curriculum_threshold)
        self._env.randomize_objects = curriculum_randomize

        base_obs = self._env.reset(rng=self._np_random)
        self._step_count = 0
        self._active_arm = "right"
        
        # Reset grasp state
        self._grasp_state = GraspState.REACHING
        self._grip_level = 0.0
        self._object_grasped = False
        self._grasp_confirmed_steps = 0
        self._cube_lifted = False
        self._prev_cube_pos = None
        self._prev_ee_pos = None
        self._env.reach_only = self.reach_only
        
        # Reset failure tracking
        self._consecutive_failed_grasps = 0
        self._steps_since_last_grasp_attempt = self._min_steps_between_attempts  # Allow immediate attempt
        
        self._capture_home_ctrl()
        if self._auto_select_arm:
            self._select_active_arm()
        # Initialize progress caches
        reach_dist = self._distance_to_cube(self._active_arm)
        place_dist = self._distance_cube_to_target()
        self._prev_reach_dist = reach_dist
        self._prev_place_dist = place_dist
        self._best_reach_dist = reach_dist
        self._best_place_dist = place_dist
        self._best_reach_ctrl = self._env.ctrl_cache.copy()
        self._best_place_ctrl = self._env.ctrl_cache.copy()
        self._grasp_initiated = False
        
        obs = self._get_obs(base_obs)
        info = self._get_info()
        
        return obs, info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        # Choose active arm from action flag immediately
        self._active_arm = "right" if action[0] >= 0 else "left"

        # Determine current phase and distance before acting
        prev_phase = "place" if (self._env.reach_success and not self.reach_only) else "reach"
        prev_dist = self._distance_cube_to_target() if prev_phase == "place" else self._distance_to_cube(self._active_arm)
        prev_ctrl = self._env.ctrl_cache.copy()
        
        # Update grasp state machine (optional)
        if self._grasp_enabled:
            self._update_grasp_state()
        else:
            self._grip_level = 0.0
            self._object_grasped = False
            self._grasp_state = GraspState.REACHING
        
        # Expand action with automatic grip
        full_action = self._expand_action(action)
        
        # Step environment
        base_obs, reward, done, env_info = self._env.step(full_action)
        self._step_count += 1
        
        # Measure new progress and revert if worse
        new_phase = "place" if (self._env.reach_success and not self.reach_only) else "reach"
        new_dist = self._distance_cube_to_target() if new_phase == "place" else self._distance_to_cube(self._active_arm)
        self._track_progress(new_phase, new_dist)
        reverted_dist = new_dist
        if new_dist > prev_dist + self._progress_tolerance:
            reverted_dist = self._revert_if_worse(new_phase, prev_dist, new_dist)
            if reverted_dist == prev_dist and (new_phase != prev_phase):
                self._env.ctrl_cache = prev_ctrl.copy()
                self._env.data.ctrl[self._env.actuator_ids] = prev_ctrl
        new_dist = reverted_dist
        
        # Cache distances for next step
        if new_phase == "reach":
            self._prev_reach_dist = new_dist
        else:
            self._prev_place_dist = new_dist
        
        # Shape reward
        reward = self._shape_reward(reward, prev_dist, new_dist, new_phase)
        
        # Build observation
        obs = self._get_obs(base_obs)
        
        truncated = self._step_count >= self.max_episode_steps
        terminated = done and not truncated
        
        info = self._get_info()
        info.update(env_info)
        # Track success streak for curriculum
        if info.get("success", False):
            self._success_streak += 1
        else:
            self._success_streak = 0
        
        return obs, float(reward), terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        cube_pos, target_pos, ee_pos, _, right_ee, left_ee = self._get_positions()
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        dist_to_hand = np.linalg.norm(cube_pos - ee_pos)
        dist_ee_target_active = np.linalg.norm(ee_pos - target_pos)
        dist_ee_target_right = np.linalg.norm(right_ee - target_pos)
        dist_ee_target_left = np.linalg.norm(left_ee - target_pos)
        success_reach = dist_to_hand < self.success_threshold
        success_place = dist_to_target < self.success_threshold
        
        return {
            "step_count": self._step_count,
            "max_episode_steps": self.max_episode_steps,
            "cube_pos": cube_pos.copy(),
            "target_pos": target_pos.copy(),
            "ee_pos": ee_pos.copy(),
            "dist_cube_target": float(dist_to_target),
            "dist_cube_hand": float(dist_to_hand),
            "dist_cube_right_hand": float(np.linalg.norm(cube_pos - right_ee)),
            "dist_cube_left_hand": float(np.linalg.norm(cube_pos - left_ee)),
            "dist_ee_target": float(dist_ee_target_active),
            "dist_ee_target_right": float(dist_ee_target_right),
            "dist_ee_target_left": float(dist_ee_target_left),
            "success_reach": success_reach,
            "success_place": success_place and not self.reach_only,
            "success": success_place and not self.reach_only,
            "curriculum_randomize": bool(self._env.randomize_objects),
            "success_streak": int(self._success_streak),
            "active_arm": self._active_arm,
            "grasp_state": self._grasp_state.name,
            "grip_level": float(self._grip_level),
            "object_grasped": self._object_grasped,
            "cube_lifted": self._cube_lifted,
            "consecutive_failed_grasps": self._consecutive_failed_grasps,
            "steps_until_next_attempt": max(0, self._min_steps_between_attempts - self._steps_since_last_grasp_attempt),
        }
    
    def render(self) -> Optional[np.ndarray]:
        if self.render_mode in ["rgb_array", "human"]:
            return self._env.render(width=640, height=480)
        return None
    
    def close(self) -> None:
        pass
    
    @property
    def np_random(self) -> np.random.Generator:
        if not hasattr(self, "_np_random"):
            self._np_random = np.random.default_rng()
        return self._np_random
