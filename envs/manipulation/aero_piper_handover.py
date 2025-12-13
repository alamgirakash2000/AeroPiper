from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import mujoco
import numpy as np

from .aero_piper_base import AeroPiperBase, _SCENES_DIR


class AeroPiperHandover(AeroPiperBase):
    """Right-to-left handover task using a light ball."""

    def __init__(
        self,
        scene_xml_path: Optional[str | Path] = None,
        reward_type: str = "dense",
        success_threshold: float = 0.05,
        randomize_objects: bool = True,
        action_scale: float = 0.5,
        frame_skip: int = 5,
    ) -> None:
        self.scene_xml_path = scene_xml_path or (_SCENES_DIR / "aeropiper_handover.xml")
        self.reward_type = reward_type
        self.success_threshold = success_threshold
        self.randomize_objects = randomize_objects
        self.table_top_z = 0.63 + 0.04

        super().__init__(
            scene_xml_path=self.scene_xml_path,
            action_scale=action_scale,
            frame_skip=frame_skip,
        )

        self.ball_body_id = self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, "handover_ball")
        self.ball_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "handover_ball_site")
        self.target_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, "handover_target")
        self.ball_qpos_adr = self.model.jnt_qposadr[
            self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "handover_ball_free")
        ]
        self.right_ee_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self.ee_sites["right"])
        self.left_ee_site_id = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, self.ee_sites["left"])

    # ------------------------------------------------------------------ #
    # Core control loop
    # ------------------------------------------------------------------ #
    def reset(self, rng: Optional[np.random.Generator | int] = None) -> np.ndarray:
        rng = np.random.default_rng(rng)
        self.reset_robot_to_home()
        self._reset_ball(rng)
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
    def _reset_ball(self, rng: np.random.Generator) -> None:
        xy = np.array([0.68, -0.18]) if not self.randomize_objects else rng.uniform([0.62, -0.25], [0.72, -0.12])
        z = self.table_top_z + 0.05
        qpos = np.array([xy[0], xy[1], z, 1.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.ball_qpos_adr : self.ball_qpos_adr + 7] = qpos
        vel_adr = self.model.jnt_dofadr[self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, "handover_ball_free")]
        self.data.qvel[vel_adr : vel_adr + 6] = 0

    def _compute_reward(self):
        ball_pos = self.data.site_xpos[self.ball_site_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        left_ee = self.data.site_xpos[self.left_ee_site_id]
        right_ee = self.data.site_xpos[self.right_ee_site_id]

        dist_ball_left = np.linalg.norm(ball_pos - left_ee)
        dist_ball_target = np.linalg.norm(ball_pos - target_pos)
        dist_ball_right = np.linalg.norm(ball_pos - right_ee)

        reward = -dist_ball_target
        if self.reward_type == "dense":
            reward -= 0.1 * dist_ball_left
            reward -= 0.05 * dist_ball_right
        if self.collision_penalty and self.check_self_collision():
            reward -= self.collision_penalty

        info = {
            "dist_ball_target": float(dist_ball_target),
            "dist_ball_left": float(dist_ball_left),
            "dist_ball_right": float(dist_ball_right),
            "success": self._is_success(dist_ball_target),
        }
        return reward, info

    def _is_success(self, dist: float) -> bool:
        return dist < self.success_threshold

    def _is_done(self, info: dict) -> bool:
        return False

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        ball_pos = self.data.site_xpos[self.ball_site_id]
        target_pos = self.data.site_xpos[self.target_site_id]
        robot_obs = self._build_base_obs()
        return np.concatenate([robot_obs, ball_pos, target_pos])

    @property
    def observation_size(self) -> int:  # type: ignore[override]
        return self._get_obs().shape[0]
