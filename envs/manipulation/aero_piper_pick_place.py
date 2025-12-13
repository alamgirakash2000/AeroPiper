from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from .aero_piper_base import AeroPiperBase, _SCENES_DIR


class AeroPiperPickPlace(AeroPiperBase):
    """Pick-and-place task on the AeroPiper dual-arm setup."""

    def __init__(
        self,
        scene_xml_path: Optional[str | Path] = None,
        reward_type: str = "dense",
        success_threshold: float = 0.10,
        randomize_objects: bool = True,
        action_scale: float = 0.5,
        frame_skip: int = 5,
        drift_reset_threshold: float = 0.10,
    ) -> None:
        self.scene_xml_path = scene_xml_path or (_SCENES_DIR / "aeropiper_pick_place.xml")
        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.randomize_objects = randomize_objects
        self.table_top_z = 0.63 + 0.04  # from the base table definition
        self.drift_reset_threshold = drift_reset_threshold

        super().__init__(
            scene_xml_path=self.scene_xml_path,
            action_scale=action_scale,
            frame_skip=frame_skip,
        )

        self.cube_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self.target_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "cube_target")
        self.cube_qpos_adr = self.model.jnt_qposadr[
            self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        ]
        # Phase tracking
        self.reach_success = False
        self.initial_cube_pos: Optional[np.ndarray] = None
        self.reach_only: bool = False

    # ------------------------------------------------------------------ #
    # Core control loop
    # ------------------------------------------------------------------ #
    def reset(self, rng: Optional[np.random.Generator | int] = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        self.reset_robot_to_home()
        self._reset_cube(rng)
        mujoco.mj_forward(self.model, self.data)
        self.reach_success = False
        self.initial_cube_pos = np.copy(self.data.site_xpos[self.cube_site_id])
        return self._get_obs()

    def step(self, action: Sequence[float]):
        self.apply_action(action)
        self.step_physics()
        reward, info = self._compute_reward()
        done = self._is_done(info)
        obs = self._get_obs()
        return obs, reward, done, info

    # ------------------------------------------------------------------ #
    # Task specifics
    # ------------------------------------------------------------------ #
    def _reset_cube(self, rng: np.random.Generator) -> None:
        # Cube spawns on table
        if self.randomize_objects:
            # Random position: left (y>0) or right (y<0) side of table
            cube_on_left = rng.choice([True, False])
            if cube_on_left:
                cube_y = rng.uniform(0.05, 0.25)  # Left side (positive y)
            else:
                cube_y = rng.uniform(-0.25, -0.05)  # Right side (negative y)
            cube_x = rng.uniform(0.60, 0.85)
            xy = np.array([cube_x, cube_y])
        else:
            xy = np.array([0.72, -0.05])  # Fixed position
        z = self.table_top_z + 0.025
        qpos = np.array([xy[0], xy[1], z, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.cube_qpos_adr : self.cube_qpos_adr + 7] = qpos
        vel_adr = self.model.jnt_dofadr[self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "cube_free")]
        self.data.qvel[vel_adr : vel_adr + 6] = 0
        
        # Also randomize target position
        if self.randomize_objects:
            # Target on opposite side from cube
            if cube_on_left:
                target_y = rng.uniform(-0.35, -0.10)  # Right side
            else:
                target_y = rng.uniform(0.10, 0.35)  # Left side
            target_x = rng.uniform(0.55, 0.75)
            target_z = self.table_top_z + 0.02
            self.model.site_pos[self.target_site_id] = [target_x, target_y, target_z]
        else:
            # Fixed target position
            self.model.site_pos[self.target_site_id] = [0.60, -0.30, 0.70]

    def _compute_reward(self):
        cube_pos = self.data.site_xpos[self.cube_site_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        right_ee = self.data.site_xpos[self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self.ee_sites["right"])]
        left_ee = self.data.site_xpos[self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self.ee_sites["left"])]
        dist_to_target = np.linalg.norm(cube_pos - target_pos)
        dist_to_right = np.linalg.norm(cube_pos - right_ee)
        dist_to_left = np.linalg.norm(cube_pos - left_ee)
        dist_to_hand = min(dist_to_right, dist_to_left)

        # Phase 1: reach cube within 10 cm, Phase 2: cube near target within 10 cm
        success_reach = dist_to_hand <= 0.10
        if success_reach and not self.reach_only:
            self.reach_success = True
        success_place = self.reach_success and (dist_to_target <= 0.10) and (not self.reach_only)

        if self.reach_only or not self.reach_success:
            # Reward reaching the cube first
            reward = -dist_to_hand
            if success_reach:
                reward += 0.5
        else:
            # After reach, drive cube to target
            reward = -dist_to_target
            if success_place:
                reward += 1.0

        if self.collision_penalty and self.check_self_collision():
            reward -= self.collision_penalty

        info = {
            "dist_cube_target": float(dist_to_target),
            "dist_cube_right_hand": float(dist_to_right),
            "dist_cube_left_hand": float(dist_to_left),
            "dist_cube_hand": float(dist_to_hand),
            "success_reach": bool(self.reach_success),
            "success_place": success_place,
            "success": success_place,
        }
        return reward, info

    def _is_success(self, dist: float) -> bool:
        return self.reach_success and (dist < self.success_threshold)

    def _is_done(self, info: dict) -> bool:
        # Relaxed drift guard: only early reset if cube drifts very far before success
        if self.initial_cube_pos is not None and not info.get("success", False):
            cube_pos = self.data.site_xpos[self.cube_site_id]
            if np.linalg.norm(cube_pos - self.initial_cube_pos) > self.drift_reset_threshold:
                info["reset_reason"] = "cube_moved"
                return True
        return False  # episodic termination otherwise handled externally

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        cube_pos = self.data.site_xpos[self.cube_site_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        robot_obs = self._build_base_obs()
        return np.concatenate([robot_obs, cube_pos, target_pos])

    @property
    def observation_size(self) -> int:  # type: ignore[override]
        return self._get_obs().shape[0]
