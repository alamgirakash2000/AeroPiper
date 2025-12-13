from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import mujoco
import numpy as np


_SCENES_DIR = Path(__file__).resolve().parents[2] / "scenes"
DEFAULT_SCENE = _SCENES_DIR / "aeropiper_base_scene.xml"


class AeroPiperBase:
    """Base wrapper around the AeroPiper dual-arm robot model."""

    def __init__(
        self,
        scene_xml_path: Optional[str | Path] = None,
        action_scale: float = 0.2,
        frame_skip: int = 5,
        collision_penalty: float = 0.0,
        normalize_actions: bool = True,
    ) -> None:
        self.scene_xml_path = self._resolve_scene_path(scene_xml_path)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml_path))
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.action_scale = action_scale
        self.collision_penalty = collision_penalty
        self.normalize_actions = normalize_actions

        # Actuator layout: 13 per arm-hand stack = 26 total.
        self.right_arm_actuators = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.right_arm_joints = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.right_hand_actuators = [
            "right_thumb_A_cmc_abd",
            "right_th1_A_tendon",
            "right_th2_A_tendon",
            "right_index_A_tendon",
            "right_middle_A_tendon",
            "right_ring_A_tendon",
            "right_pinky_A_tendon",
        ]
        self.left_arm_actuators = [
            "left_joint1",
            "left_joint2",
            "left_joint3",
            "left_joint4",
            "left_joint5",
            "left_joint6",
        ]
        self.left_arm_joints = [
            "left_joint1",
            "left_joint2",
            "left_joint3",
            "left_joint4",
            "left_joint5",
            "left_joint6",
        ]
        self.left_hand_actuators = [
            "left_thumb_A_cmc_abd",
            "left_th1_A_tendon",
            "left_th2_A_tendon",
            "left_index_A_tendon",
            "left_middle_A_tendon",
            "left_ring_A_tendon",
            "left_pinky_A_tendon",
        ]

        self.actuator_names: Sequence[str] = (
            self.right_arm_actuators
            + self.right_hand_actuators
            + self.left_arm_actuators
            + self.left_hand_actuators
        )
        self.actuator_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names],
            dtype=np.int32,
        )
        self.arm_joint_ids = np.array(
            [self._name_to_id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.right_arm_joints + self.left_arm_joints],
            dtype=np.int32,
        )
        self.arm_qadr = np.array([self.model.jnt_qposadr[jid] for jid in self.arm_joint_ids], dtype=np.int32)

        self.ee_sites = {"right": "grasp_site", "left": "left_grasp_site"}
        self.fingertip_sites = [
            "if_tip",
            "mf_tip",
            "rf_tip",
            "pf_tip",
            "th_tip",
            "left_if_tip",
            "left_mf_tip",
            "left_rf_tip",
            "left_pf_tip",
            "left_th_tip",
        ]
        self.body_ids_left = self._collect_body_ids(prefixes=["left_"])
        self.body_ids_right = self._collect_body_ids(
            prefixes=[
                "right_",
                "palm",
                "index_",
                "middle_",
                "ring_",
                "pinky_",
                "thumb_",
                "t_link",
            ],
            exact={"palm", "right_base_link", "right_link1", "right_link2", "right_link3", "right_link4", "right_link5", "right_link6"},
        )

        self.home_key_id = self._find_keyframe("home")

    # --------------------------------------------------------------------- #
    # Utility helpers
    # --------------------------------------------------------------------- #
    def _resolve_scene_path(self, scene_xml_path: Optional[str | Path]) -> Path:
        if scene_xml_path is None:
            return DEFAULT_SCENE
        path = Path(scene_xml_path)
        if path.is_absolute():
            return path
        return _SCENES_DIR / path

    def _name_to_id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        try:
            return mujoco.mj_name2id(self.model, obj_type, name)
        except KeyError:
            raise KeyError(
                f"Name '{name}' not found in model (type={obj_type.name}). "
                f"Did you load the correct scene? ({self.scene_xml_path})"
            ) from None

    def _collect_body_ids(self, prefixes: Optional[Iterable[str]] = None, exact: Optional[Iterable[str]] = None) -> set[int]:
        ids: set[int] = set()
        for name in exact or []:
            ids.add(self._name_to_id(mujoco.mjtObj.mjOBJ_BODY, name))

        if prefixes:
            for idx in range(self.model.nbody):
                name = self.model.body(idx).name
                if any(name.startswith(prefix) for prefix in prefixes):
                    ids.add(idx)
        return ids

    def _find_keyframe(self, name: str) -> Optional[int]:
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, name)
        except KeyError:
            return None

    # --------------------------------------------------------------------- #
    # Robot-centric utilities
    # --------------------------------------------------------------------- #
    @property
    def action_size(self) -> int:
        return len(self.actuator_names)

    def reset_robot_to_home(self) -> mujoco.MjData:
        if self.home_key_id is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        # Initialize controller targets to hold the current pose:
        # - For joint-driven actuators, set ctrl to current joint qpos
        # - For tendon-driven actuators, set ctrl to current tendon length
        for act_id in range(self.model.nu):
            trn_type = self.model.actuator_trntype[act_id]
            trn_id = self.model.actuator_trnid[act_id][0]
            if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                qadr = self.model.jnt_qposadr[trn_id]
                self.data.ctrl[act_id] = self.data.qpos[qadr]
            elif trn_type == mujoco.mjtTrn.mjTRN_TENDON:
                self.data.ctrl[act_id] = self.data.ten_length[trn_id]

        self.ctrl_cache = np.copy(self.data.ctrl)
        return self.data

    def apply_action(self, action: Sequence[float]) -> mujoco.MjData:
        action_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if action_arr.shape[0] != self.action_size:
            raise ValueError(f"Expected {self.action_size} actions, got {action_arr.shape[0]}")

        # Base ctrl ranges from the model
        ctrl_range = self.model.actuator_ctrlrange[self.actuator_ids]
        low = ctrl_range[:, 0].copy()
        high = ctrl_range[:, 1].copy()

        # Hardcoded arm joint limits (radians), overriding XML ctrlrange for arms only.
        # Order of actuator_names: right_arm(0-5), right_hand(6-12), left_arm(13-18), left_hand(19-25)
        right_arm_low = np.array([-0.85, 1.0, -2.697, -1.5, -1.22, -3.14], dtype=np.float64)
        left_arm_low = np.array([-1.57, 1.0, -2.697, 0, -1.22, -3.14], dtype=np.float64)
        right_arm_high = np.array([1.57, 2.75, -0.5, 0, 1.22, 3.14], dtype=np.float64)
        left_arm_high = np.array([0.85, 2.75, -0.5, 1.5, 1.22, 3.14], dtype=np.float64)
        # Right arm indices 0-5
        low[0:6] = right_arm_low
        high[0:6] = right_arm_high
        # Left arm indices 13-18
        low[13:19] = left_arm_low
        high[13:19] = left_arm_high

        if self.normalize_actions:
            action_arr = np.clip(action_arr, -1.0, 1.0)

        # Hold-current semantics: zero action keeps the last ctrl targets.
        delta = self.action_scale * action_arr
        if not hasattr(self, "ctrl_cache"):
            self.ctrl_cache = np.copy(self.data.ctrl)
        proposed = self.ctrl_cache + delta
        scaled = np.clip(proposed, low, high)
        self.data.ctrl[self.actuator_ids] = scaled
        self.ctrl_cache = scaled.copy()
        return self.data

    def get_ee_positions(self) -> Dict[str, np.ndarray]:
        return {
            arm: np.copy(self.data.site_xpos[self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, site)])
            for arm, site in self.ee_sites.items()
        }

    def get_fingertip_positions(self) -> Dict[str, np.ndarray]:
        positions: Dict[str, np.ndarray] = {}
        for site in self.fingertip_sites:
            try:
                sid = self._name_to_id(mujoco.mjtObj.mjOBJ_SITE, site)
                positions[site] = np.copy(self.data.site_xpos[sid])
            except KeyError:
                # Skip if a fingertip is missing in the loaded variant.
                continue
        return positions

    def get_robot_obs(self) -> np.ndarray:
        ee = self.get_ee_positions()
        ee_flat = np.concatenate([ee["right"], ee["left"]])
        return np.concatenate([self.data.qpos, self.data.qvel, ee_flat])

    def check_self_collision(self) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if (body1 in self.body_ids_left and body2 in self.body_ids_right) or (
                body2 in self.body_ids_left and body1 in self.body_ids_right
            ):
                return True
        return False

    def pipeline_step(self) -> mujoco.MjData:
        mujoco.mj_step(self.model, self.data)
        return self.data

    def step_physics(self) -> mujoco.MjData:
        for _ in range(self.frame_skip):
            # Hard clamp arm joints to their current ctrl target to eliminate drift.
            for act_id in self.actuator_ids:
                trn_type = self.model.actuator_trntype[act_id]
                trn_id = self.model.actuator_trnid[act_id][0]
                if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
                    qadr = self.model.jnt_qposadr[trn_id]
                    self.data.qpos[qadr] = self.data.ctrl[act_id]
                    self.data.qvel[qadr] = 0.0
            self.pipeline_step()
        return self.data

    # --------------------------------------------------------------------- #
    # Observation helpers
    # --------------------------------------------------------------------- #
    def observation_size(self) -> int:
        return self.get_robot_obs().shape[0]

    def _build_base_obs(self) -> np.ndarray:
        return self.get_robot_obs()

    # --------------------------------------------------------------------- #
    # Rendering helper
    # --------------------------------------------------------------------- #
    def render(self, camera: str | None = None, width: int = 640, height: int = 480) -> np.ndarray:
        viewport = mujoco.MjrRect(0, 0, width, height)
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        cam = mujoco.MjvCamera()
        option = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(cam)
        if camera is not None:
            try:
                cam.fixedcamid = self._name_to_id(mujoco.mjtObj.mjOBJ_CAMERA, camera)
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            except KeyError:
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjv_updateScene(self.model, self.data, option, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_buffer, None, viewport, context)
        return rgb_buffer
