"""
Real-time upper-body tracking using Ultralytics YOLO pose models to drive the AeroPiper arm.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "OpenCV (cv2) is required for webcam arm tracking. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install opencv-python`."
    ) from exc

try:
    import torch  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "PyTorch is required for running the YOLO pose estimator. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`."
    ) from exc

try:
    from ultralytics import YOLO  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "Ultralytics YOLO is required for upper-body pose tracking. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install ultralytics`."
    ) from exc


ARM_LABELS = [
    "BaseYaw",
    "ShoulderPitch",
    "ShoulderRoll",
    "ElbowFlex",
    "WristPitch",
    "WristRoll",
]

YOLO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_CONNECTIONS = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

DEFAULT_YOLO_MODEL = "yolov8n-pose.pt"
DEFAULT_YOLO_IMGSZ = 640


class COCOKeypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


@dataclass
class PoseSample:
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox: Optional[np.ndarray]
    image_shape: Tuple[int, int]


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return np.zeros_like(vec)
    return vec / norm


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_u = _normalize(v1)
    v2_u = _normalize(v2)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return float(np.arccos(dot))


class ArmGestureController:
    """
    Track the user's left arm with a webcam and emit six normalized DOFs in [-1, 1]:

    [base_yaw, shoulder_pitch, shoulder_roll, elbow_flex, wrist_pitch, wrist_roll]
    """

    def __init__(
        self,
        camera_index: int = 0,
        show_preview: bool = True,
        mirror_preview: bool = True,
        smoothing: float = 0.5,
        idle_decay: float = 0.96,
        visibility_threshold: float = 0.45,
        max_step: float = 0.35,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        yolo_weights: str = DEFAULT_YOLO_MODEL,
        yolo_confidence: float = 0.45,
        yolo_iou: float = 0.5,
        yolo_imgsz: int = DEFAULT_YOLO_IMGSZ,
        yolo_max_det: int = 1,
        yolo_device: Optional[str] = None,
        yolo_frame_skip: int = 0,
        yolo_half_precision: bool = False,
    ):
        self._camera_index = camera_index
        self._show_preview = show_preview
        self._mirror_preview = mirror_preview
        self._alpha = float(np.clip(smoothing, 0.0, 1.0))
        self._idle_decay = float(np.clip(idle_decay, 0.80, 0.999))
        self._visibility_threshold = float(np.clip(visibility_threshold, 0.05, 0.99))
        self._max_step = float(np.clip(max_step, 0.05, 1.5))
        self._preview_window = "AeroPiper Left Arm"

        self._frame_provider = frame_provider
        self._owns_capture = frame_provider is None
        self._capture = self._open_camera(camera_index) if self._owns_capture else None

        self._device = self._resolve_device(yolo_device)
        self._yolo_conf = float(np.clip(yolo_confidence, 0.05, 0.95))
        self._yolo_iou = float(np.clip(yolo_iou, 0.1, 0.9))
        self._yolo_imgsz = int(max(256, yolo_imgsz))
        self._yolo_max_det = int(max(1, yolo_max_det))
        self._frame_skip = max(0, int(yolo_frame_skip))
        self._frame_stride = self._frame_skip + 1
        self._frame_counter = 0

        self._use_half = False
        if yolo_half_precision:
            if str(self._device).startswith("cuda"):
                self._use_half = True
            else:
                print(
                    "[ArmGestureController] Ignoring half precision request because CUDA is unavailable; running in full precision."
                )

        try:
            self._pose_model = YOLO(yolo_weights)
        except Exception as exc:  # pragma: no cover - handled at runtime
            raise RuntimeError(
                f"Failed to load YOLO pose weights '{yolo_weights}'. "
                "Ensure the file exists or that the machine can download it automatically."
            ) from exc
        self._pose_model.to(self._device)
        if self._use_half:
            self._pose_model.model.half()
        self._pose_kwargs = {
            "conf": self._yolo_conf,
            "iou": self._yolo_iou,
            "imgsz": self._yolo_imgsz,
            "max_det": self._yolo_max_det,
            "device": self._device,
            "verbose": False,
            "half": self._use_half,
        }

        self._arm = np.zeros(len(ARM_LABELS), dtype=np.float32)
        self._initialized = False
        self._stop_requested = False
        self._last_detection_time = 0.0
        self._last_measurement: Optional[np.ndarray] = None
        self._last_landmarks: Optional[np.ndarray] = None
        self._elbow_raw_min: Optional[float] = None
        self._elbow_raw_max: Optional[float] = None
        self._elbow_calibration_samples = 0

        self._last_keypoints_xy: Optional[np.ndarray] = None
        self._last_keypoints_conf: Optional[np.ndarray] = None
        self._last_bbox: Optional[np.ndarray] = None

        self._segment_lengths = {
            "upper": 0.55,
            "forearm": 0.60,
        }

    def _resolve_device(self, requested: Optional[str]) -> str:
        if requested:
            if requested.startswith("cuda") and not torch.cuda.is_available():
                print(
                    f"[ArmGestureController] Requested {requested} but CUDA is unavailable; falling back to CPU."
                )
                return "cpu"
            return requested
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def _open_camera(self, index: int):
        if os.name == "nt":
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open webcam index {index}. "
                "Activate the 'aeropiper' conda environment and ensure a camera is connected."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        capture.set(cv2.CAP_PROP_FPS, 30)
        return capture

    def update(self) -> np.ndarray:
        """Read the current frame, update pose estimates, and return 6 normalized DOFs."""

        if self._stop_requested:
            return self._arm.copy()

        frame = self._read_frame()
        if frame is None:
            self._stop_requested = True
            return self._arm.copy()

        stride_hit = (self._frame_counter % self._frame_stride) == 0
        self._frame_counter += 1

        measurement: Optional[np.ndarray] = None
        new_observation = False

        if stride_hit:
            measurement = self._extract_measurement(frame)
            new_observation = measurement is not None

        now = time.time()

        if new_observation and measurement is not None:
            measurement = self._limit_step(measurement)
            if not self._initialized:
                self._arm = measurement
                self._initialized = True
            else:
                self._arm = (
                    self._alpha * measurement + (1.0 - self._alpha) * self._arm
                )
            self._last_detection_time = now
        elif self._initialized and now - self._last_detection_time > 0.3:
            self._arm *= self._idle_decay
            if self._last_measurement is not None:
                self._last_measurement *= self._idle_decay

        if self._show_preview:
            self._render_preview(frame)

        return self._arm.copy()

    def should_stop(self) -> bool:
        return self._stop_requested

    def close(self):
        if self._owns_capture and self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._show_preview:
            try:
                cv2.destroyWindow(self._preview_window)
            except cv2.error:
                pass

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._frame_provider is not None:
            frame = self._frame_provider()
            return frame
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def _extract_measurement(
        self, frame: np.ndarray
    ) -> Optional[np.ndarray]:
        sample = self._infer_pose(frame)
        if sample is None:
            self._last_landmarks = None
            self._last_keypoints_xy = None
            self._last_keypoints_conf = None
            self._last_bbox = None
            return None

        self._last_keypoints_xy = sample.keypoints_xy
        self._last_keypoints_conf = sample.keypoints_conf
        self._last_bbox = sample.bbox

        coords = self._build_coords(sample)
        self._last_landmarks = coords
        measurement = self._landmarks_to_arm(coords)
        return measurement

    def _infer_pose(self, frame: np.ndarray) -> Optional[PoseSample]:
        try:
            results = self._pose_model.predict(frame, **self._pose_kwargs)
        except RuntimeError as exc:  # pragma: no cover - handled at runtime
            print(f"[ArmGestureController] YOLO inference failed: {exc}")
            return None

        if not results:
            return None
        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or keypoints.xy.shape[0] == 0:
            return None

        boxes = getattr(result, "boxes", None)
        idx = 0
        if boxes is not None and boxes.conf is not None and boxes.conf.numel() > 0:
            idx = int(torch.argmax(boxes.conf).item())

        xy = keypoints.xy[idx].detach().cpu().numpy().astype(np.float32)
        conf = (
            keypoints.conf[idx].detach().cpu().numpy().astype(np.float32)
            if keypoints.conf is not None
            else np.ones(xy.shape[0], dtype=np.float32)
        )
        bbox = None
        if boxes is not None and boxes.xyxy is not None and boxes.xyxy.shape[0] > idx:
            bbox = boxes.xyxy[idx].detach().cpu().numpy().astype(np.float32)

        h, w = frame.shape[:2]
        return PoseSample(xy, conf, bbox, (h, w))

    def _build_coords(self, sample: PoseSample) -> np.ndarray:
        count = min(len(YOLO_KEYPOINT_NAMES), sample.keypoints_xy.shape[0])
        coords = np.zeros((len(YOLO_KEYPOINT_NAMES), 4), dtype=np.float32)
        if count == 0:
            return coords

        h, w = sample.image_shape
        inv_w = 1.0 / max(w, 1)
        inv_h = 1.0 / max(h, 1)

        coords[:count, 0] = sample.keypoints_xy[:count, 0] * inv_w
        coords[:count, 1] = sample.keypoints_xy[:count, 1] * inv_h
        coords[:count, 1] = 1.0 - coords[:count, 1]
        coords[:count, 3] = sample.keypoints_conf[:count]
        return coords

    def _update_depth_estimates(self, coords: np.ndarray, scale: float):
        kp = COCOKeypoint
        coords[kp.LEFT_SHOULDER, 2] = 0.0
        coords[kp.RIGHT_SHOULDER, 2] = 0.0
        coords[kp.LEFT_HIP, 2] = 0.0

        if scale < 1e-4:
            return

        def _apply(origin_idx: int, target_idx: int, key: str):
            origin = coords[origin_idx]
            target = coords[target_idx]
            if (
                origin[3] < self._visibility_threshold
                or target[3] < self._visibility_threshold
            ):
                coords[target_idx, 2] = coords[origin_idx, 2]
                return
            vec = (target[:2] - origin[:2]) / scale
            obs_len = float(np.linalg.norm(vec))
            stored = self._segment_lengths[key]
            if obs_len > stored:
                stored = obs_len
                self._segment_lengths[key] = stored
            span = max(stored**2 - obs_len**2, 0.0)
            depth = np.sqrt(span) * scale
            coords[target_idx, 2] = coords[origin_idx, 2] + depth

        _apply(kp.LEFT_SHOULDER, kp.LEFT_ELBOW, "upper")
        _apply(kp.LEFT_ELBOW, kp.LEFT_WRIST, "forearm")

    def _landmarks_to_arm(self, coords: np.ndarray) -> Optional[np.ndarray]:
        kp = COCOKeypoint
        required = [
            kp.LEFT_SHOULDER,
            kp.RIGHT_SHOULDER,
            kp.LEFT_ELBOW,
            kp.LEFT_WRIST,
            kp.LEFT_HIP,
        ]
        visibilities = coords[:, 3]
        for landmark in required:
            if visibilities[landmark.value] < self._visibility_threshold:
                return None

        left_shoulder = coords[kp.LEFT_SHOULDER, :3].copy()
        right_shoulder = coords[kp.RIGHT_SHOULDER, :3].copy()
        left_elbow = coords[kp.LEFT_ELBOW, :3].copy()
        left_wrist = coords[kp.LEFT_WRIST, :3].copy()
        left_hip = coords[kp.LEFT_HIP, :3].copy()

        shoulder_width = np.linalg.norm(left_shoulder[:2] - right_shoulder[:2])
        scale = max(shoulder_width, 1e-3)
        self._update_depth_estimates(coords, scale)

        left_shoulder = coords[kp.LEFT_SHOULDER, :3]
        right_shoulder = coords[kp.RIGHT_SHOULDER, :3]
        left_elbow = coords[kp.LEFT_ELBOW, :3]
        left_wrist = coords[kp.LEFT_WRIST, :3]
        left_hip = coords[kp.LEFT_HIP, :3]

        upper = (left_elbow - left_shoulder) / scale
        forearm = (left_wrist - left_elbow) / scale
        wrist_from_shoulder = (left_wrist - left_shoulder) / scale

        if np.linalg.norm(forearm) < 1e-4 or np.linalg.norm(upper) < 1e-4:
            return None

        base_yaw = np.arctan2(wrist_from_shoulder[0], wrist_from_shoulder[2] + 1e-5)
        shoulder_pitch = np.arctan2(
            wrist_from_shoulder[1], np.linalg.norm(wrist_from_shoulder[[0, 2]]) + 1e-5
        )
        shoulder_roll = np.arctan2(
            wrist_from_shoulder[0],
            np.linalg.norm([wrist_from_shoulder[1], wrist_from_shoulder[2]]) + 1e-5,
        )

        elbow_angle = _angle_between(-upper, forearm)
        elbow_raw = 2.0 * ((np.pi - elbow_angle) / np.pi) - 1.0
        elbow_flex = self._normalize_elbow(elbow_raw)

        forearm_unit = _normalize(forearm)
        shoulder_axis = _normalize(right_shoulder - left_shoulder)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        torso_forward = np.cross(shoulder_axis, world_up)
        if np.linalg.norm(torso_forward) < 1e-5:
            torso_forward = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        torso_forward = _normalize(torso_forward)

        hand_dir = np.cross(torso_forward, forearm_unit)
        if np.linalg.norm(hand_dir) < 1e-5:
            hand_dir = torso_forward
        hand_dir = _normalize(hand_dir)

        palm_perp = np.cross(forearm_unit, hand_dir)
        if np.linalg.norm(palm_perp) < 1e-5:
            palm_perp = np.cross(forearm_unit, torso_forward)
        palm_perp = _normalize(palm_perp)

        ref = world_up - np.dot(world_up, forearm_unit) * forearm_unit
        if np.linalg.norm(ref) < 1e-5:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ref = ref - np.dot(ref, forearm_unit) * forearm_unit
        ref = _normalize(ref)

        wrist_pitch = float(np.clip(np.dot(hand_dir, _normalize(-forearm)), -1.0, 1.0))
        wrist_roll = np.arctan2(
            np.dot(np.cross(ref, palm_perp), forearm_unit),
            np.dot(ref, palm_perp) + 1e-5,
        )

        values = np.asarray(
            [
                base_yaw / np.deg2rad(100.0),
                shoulder_pitch / np.deg2rad(100.0),
                shoulder_roll / np.deg2rad(80.0),
                elbow_flex,
                wrist_pitch,
                wrist_roll / np.deg2rad(135.0),
            ],
            dtype=np.float32,
        )

        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(values, -1.0, 1.0)

    def _limit_step(self, measurement: np.ndarray) -> np.ndarray:
        limited = np.asarray(measurement, dtype=np.float32)
        if self._last_measurement is None:
            self._last_measurement = limited
            return limited
        delta = limited - self._last_measurement
        bounded = self._last_measurement + np.clip(delta, -self._max_step, self._max_step)
        self._last_measurement = bounded
        return bounded

    def _render_preview(self, frame):
        if frame is None or not self._show_preview:
            return
        overlay = frame.copy()
        self.annotate_frame(overlay)
        if self._mirror_preview:
            overlay = cv2.flip(overlay, 1)
        stats = " ".join(f"{name}:{val:+4.2f}" for name, val in zip(ARM_LABELS, self._arm))
        cv2.putText(
            overlay,
            "Left arm tracking (Q/ESC exit)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            stats,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        try:
            cv2.imshow(self._preview_window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._stop_requested = True
        except cv2.error:
            print("OpenCV preview unavailable, continuing without it.")
            self._show_preview = False

    def annotate_frame(self, frame: Optional[np.ndarray]):
        if (
            frame is None
            or self._last_keypoints_xy is None
            or self._last_keypoints_conf is None
        ):
            return

        for idx_a, idx_b in COCO_CONNECTIONS:
            if (
                idx_a >= len(self._last_keypoints_xy)
                or idx_b >= len(self._last_keypoints_xy)
            ):
                continue
            if (
                self._last_keypoints_conf[idx_a] < self._visibility_threshold
                or self._last_keypoints_conf[idx_b] < self._visibility_threshold
            ):
                continue
            pt1 = (
                int(self._last_keypoints_xy[idx_a, 0]),
                int(self._last_keypoints_xy[idx_a, 1]),
            )
            pt2 = (
                int(self._last_keypoints_xy[idx_b, 0]),
                int(self._last_keypoints_xy[idx_b, 1]),
            )
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

        for idx, conf in enumerate(self._last_keypoints_conf):
            if conf < self._visibility_threshold or idx >= len(self._last_keypoints_xy):
                continue
            pt = (
                int(self._last_keypoints_xy[idx, 0]),
                int(self._last_keypoints_xy[idx, 1]),
            )
            cv2.circle(frame, pt, 4, (0, 255, 0), -1)

        if self._last_bbox is not None and len(self._last_bbox) == 4:
            x1, y1, x2, y2 = self._last_bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 128, 255), 2)

    def _normalize_elbow(self, raw_value: float) -> float:
        if self._elbow_calibration_samples < 200:
            if self._elbow_raw_min is None or raw_value < self._elbow_raw_min:
                self._elbow_raw_min = raw_value
            if self._elbow_raw_max is None or raw_value > self._elbow_raw_max:
                self._elbow_raw_max = raw_value
            self._elbow_calibration_samples += 1

        if (
            self._elbow_raw_min is not None
            and self._elbow_raw_max is not None
            and (self._elbow_raw_max - self._elbow_raw_min) > 1e-3
        ):
            span = self._elbow_raw_max - self._elbow_raw_min
            normalized = (raw_value - self._elbow_raw_min) / span
            return float(np.clip(normalized * 2.0 - 1.0, -1.0, 1.0))

        return float(np.clip(raw_value, -1.0, 1.0))

