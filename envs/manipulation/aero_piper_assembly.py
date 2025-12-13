from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from .aero_piper_base import AeroPiperBase, _SCENES_DIR


class AeroPiperAssembly(AeroPiperBase):
    """Peg-in-hole assembly task."""

    def __init__(
        self,
        scene_xml_path: Optional[str | Path] = None,
        reward_type: str = "dense",
        success_threshold: float = 0.015,
        randomize_objects: bool = True,
        action_scale: float = 0.5,
        frame_skip: int = 5,
    ) -> None:
        self.scene_xml_path = scene_xml_path or (_SCENES_DIR / "aeropiper_assembly.xml")
        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.randomize_objects = randomize_objects
        self.table_top_z = 0.63 + 0.04

        super().__init__(
            scene_xml_path=self.scene_xml_path,
            action_scale=action_scale,
            frame_skip=frame_skip,
        )

        self.peg_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, "peg")
        self.peg_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "peg_site")
        self.target_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "assembly_target")
        self.peg_qpos_adr = self.model.jnt_qposadr[self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "peg_free")]

    # ------------------------------------------------------------------ #
    # Core control loop
    # ------------------------------------------------------------------ #
    def reset(self, rng: Optional[np.random.Generator | int] = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        self.reset_robot_to_home()
        self._reset_peg(rng)
        mujoco.mj_forward(self.model, self.data)
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
    def _reset_peg(self, rng: np.random.Generator) -> None:
        xy = np.array([0.78, -0.12]) if not self.randomize_objects else rng.uniform([0.70, -0.18], [0.86, -0.05])
        z = self.table_top_z + 0.08
        qpos = np.array([xy[0], xy[1], z, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.peg_qpos_adr : self.peg_qpos_adr + 7] = qpos
        vel_adr = self.model.jnt_dofadr[self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "peg_free")]
        self.data.qvel[vel_adr : vel_adr + 6] = 0

    def _compute_reward(self):
        peg_pos = self.data.site_xpos[self.peg_site_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        dist = np.linalg.norm(peg_pos - target_pos)
        reward = -dist
        if self.collision_penalty and self.check_self_collision():
            reward -= self.collision_penalty
        info = {"dist_peg_target": float(dist), "success": self._is_success(dist)}
        return reward, info

    def _is_success(self, dist: float) -> bool:
        return dist < self.success_threshold

    def _is_done(self, info: dict) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        peg_pos = self.data.site_xpos[self.peg_site_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        robot_obs = self._build_base_obs()
        return np.concatenate([robot_obs, peg_pos, target_pos])

    @property
    def observation_size(self) -> int:  # type: ignore[override]
        return self._get_obs().shape[0]
