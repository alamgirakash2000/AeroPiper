"""
Simplified Pick-and-Place Environment Wrapper
==============================================

Reduced action space for easier learning:
- 8 actions total: [arm_select, j1, j2, j3, j4, j5, j6, finger_grip]
- arm_select < 0: control left arm, >= 0: control right arm
- finger_grip: single value controls all fingers uniformly

Original action space: 26 DOF (13 per arm)
Simplified action space: 8 DOF (arm selection + 6 joints + 1 grip)

Key features:
- Only selected arm moves, other arm stays at HOME position
- Grip value is absolute (not delta): -1=open, 0=half, +1=closed
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.manipulation import AeroPiperPickPlace


class AeroPiperPickPlaceSimplified(gym.Env):
    """
    Simplified pick-and-place with reduced action space.
    
    Action Space (8D):
    ------------------
    [0] arm_select: < 0 = left arm, >= 0 = right arm
    [1-6] joint1-6: 6 DOF arm control (delta)
    [7] finger_grip: absolute position for all fingers (-1=open, 0=half, +1=closed)
    
    The inactive arm returns to HOME/rest position.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        max_episode_steps: int = 500,
        reward_type: str = "dense",
        success_threshold: float = 0.04,
        randomize_objects: bool = True,
        action_scale: float = 0.5,
        frame_skip: int = 5,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0
        self.success_threshold = success_threshold
        self._action_scale = action_scale
        
        # Create underlying environment
        self._env = AeroPiperPickPlace(
            reward_type=reward_type,
            success_threshold=success_threshold,
            randomize_objects=randomize_objects,
            action_scale=action_scale,
            frame_skip=frame_skip,
        )
        
        # Simplified action space: 8 dimensions
        # [arm_select, j1, j2, j3, j4, j5, j6, finger_grip]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )
        
        # Observation space from underlying env
        sample_obs = self._env.reset()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=sample_obs.shape,
            dtype=np.float32,
        )
        
        # Store model/data references
        self.model = self._env.model
        self.data = self._env.data
        
        # Actuator indices for mapping
        # Right arm: actuators 0-5 (joints), 6-12 (fingers)
        # Left arm: actuators 13-18 (joints), 19-25 (fingers)
        self._right_arm_idx = list(range(0, 6))
        self._right_finger_idx = list(range(6, 13))
        self._left_arm_idx = list(range(13, 19))
        self._left_finger_idx = list(range(19, 26))
        
        # Get ctrl ranges for absolute position mapping
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        
        # Store home position ctrl values (captured after reset)
        self._home_ctrl = None
        self._active_arm = "right"
        
    def _capture_home_ctrl(self):
        """Capture the home position ctrl values after reset."""
        self._home_ctrl = self._env.ctrl_cache.copy()
        
    def _get_finger_ctrl_targets(self, grip_value: float, finger_idx: list) -> np.ndarray:
        """
        Convert grip value [-1, 1] to absolute ctrl targets for finger actuators.
        
        grip_value: -1 (fully open) to +1 (fully closed)
        finger_idx: list of actuator indices for the fingers
        
        Returns: array of absolute ctrl targets
        """
        targets = np.zeros(len(finger_idx))
        
        # Normalize grip from [-1, 1] to [0, 1] where 0=open, 1=closed
        grip_norm = (grip_value + 1.0) / 2.0
        
        for i, act_idx in enumerate(finger_idx):
            low = self._ctrl_range[act_idx, 0]
            high = self._ctrl_range[act_idx, 1]
            
            # Check if this is thumb abduction (joint) or tendon
            trn_type = self.model.actuator_trntype[act_idx]
            
            if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                # Thumb abduction: higher value = more adducted (toward palm)
                # grip=0 (open) -> low, grip=1 (closed) -> high
                targets[i] = low + grip_norm * (high - low)
            else:
                # Tendon actuator: LOWER length = more flexion
                # grip=0 (open) -> high (extended), grip=1 (closed) -> low (flexed)
                targets[i] = high - grip_norm * (high - low)
        
        return targets
    
    def _expand_action(self, simplified_action: np.ndarray) -> np.ndarray:
        """
        Expand 8D simplified action to 26D full action.
        
        For the ACTIVE arm: apply the joint deltas and set finger grip position
        For the INACTIVE arm: compute delta to return to HOME position
        """
        arm_select = simplified_action[0]
        arm_joints = simplified_action[1:7]  # 6 joint delta values
        grip = simplified_action[7]  # absolute grip position [-1, 1]
        
        # Get current ctrl values
        current_ctrl = self._env.ctrl_cache.copy()
        
        # Compute deltas to achieve desired state
        full_action = np.zeros(26, dtype=np.float64)
        
        if arm_select >= 0:
            # Control RIGHT arm
            self._active_arm = "right"
            active_arm_idx = self._right_arm_idx
            active_finger_idx = self._right_finger_idx
            inactive_arm_idx = self._left_arm_idx
            inactive_finger_idx = self._left_finger_idx
        else:
            # Control LEFT arm
            self._active_arm = "left"
            active_arm_idx = self._left_arm_idx
            active_finger_idx = self._left_finger_idx
            inactive_arm_idx = self._right_arm_idx
            inactive_finger_idx = self._right_finger_idx
        
        # === ACTIVE ARM ===
        # Arm joints: use delta control as provided
        full_action[active_arm_idx] = arm_joints
        
        # Fingers: compute delta to reach absolute grip position
        finger_targets = self._get_finger_ctrl_targets(grip, active_finger_idx)
        finger_current = current_ctrl[active_finger_idx]
        finger_delta = (finger_targets - finger_current) / self._action_scale
        finger_delta = np.clip(finger_delta, -1.0, 1.0)
        full_action[active_finger_idx] = finger_delta
        
        # === INACTIVE ARM ===
        # Compute delta to return to home position
        if self._home_ctrl is not None:
            # Arm: move toward home
            home_arm = self._home_ctrl[inactive_arm_idx]
            current_arm = current_ctrl[inactive_arm_idx]
            arm_delta = (home_arm - current_arm) / self._action_scale
            arm_delta = np.clip(arm_delta, -1.0, 1.0)
            full_action[inactive_arm_idx] = arm_delta
            
            # Fingers: move toward home (open)
            home_fingers = self._home_ctrl[inactive_finger_idx]
            current_fingers = current_ctrl[inactive_finger_idx]
            finger_home_delta = (home_fingers - current_fingers) / self._action_scale
            finger_home_delta = np.clip(finger_home_delta, -1.0, 1.0)
            full_action[inactive_finger_idx] = finger_home_delta
        else:
            # No home captured yet, just hold position
            full_action[inactive_arm_idx] = 0.0
            full_action[inactive_finger_idx] = 0.0
        
        return full_action
    
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
            
        obs = self._env.reset(rng=self._np_random)
        self._step_count = 0
        self._active_arm = "right"
        
        # Capture home position after reset
        self._capture_home_ctrl()
        
        info = self._get_info()
        return obs.astype(np.float32), info
    
    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Clip and expand action
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        full_action = self._expand_action(action)
        
        # Step underlying environment
        obs, reward, done, env_info = self._env.step(full_action)
        self._step_count += 1
        
        # Termination vs truncation
        truncated = self._step_count >= self.max_episode_steps
        terminated = done and not truncated
        
        # Build info
        info = self._get_info()
        info.update(env_info)
        info["active_arm"] = self._active_arm
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def _get_info(self) -> Dict[str, Any]:
        cube_pos = self._env.data.site_xpos[self._env.cube_site_id]
        target_pos = self._env.data.site_xpos[self._env.target_site_id]
        
        right_ee = self._env.get_ee_positions()["right"]
        left_ee = self._env.get_ee_positions()["left"]
        
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        active_ee = right_ee if self._active_arm == "right" else left_ee
        dist_to_hand = np.linalg.norm(cube_pos - active_ee)
        
        return {
            "step_count": self._step_count,
            "max_episode_steps": self.max_episode_steps,
            "cube_pos": cube_pos.copy(),
            "target_pos": target_pos.copy(),
            "right_ee_pos": right_ee.copy(),
            "left_ee_pos": left_ee.copy(),
            "dist_cube_target": float(dist_to_target),
            "dist_cube_hand": float(dist_to_hand),
            "success": dist_to_target < self.success_threshold,
            "active_arm": self._active_arm,
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
