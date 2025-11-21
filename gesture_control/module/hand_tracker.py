import os
import time
from enum import IntEnum
from typing import Callable, Optional, Tuple

try:
    import cv2  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "OpenCV (cv2) is required for webcam hand tracking. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install opencv-python`."
    ) from exc

try:
    import torch  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the YOLO-based hand tracker. "
        "Install it via `pip install torch torchvision torchaudio --index-url "
        "https://download.pytorch.org/whl/cu121` (adjust CUDA version as needed)."
    ) from exc

try:
    from ultralytics import YOLO  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "Ultralytics YOLO is required. Install it via "
        "`pip install ultralytics`."
    ) from exc

try:
    import mediapipe as mp  # type: ignore[import]
except ImportError:
    mp = None

import numpy as np

if mp is not None:
    MP_HANDS = mp.solutions.hands
    MP_DRAW = mp.solutions.drawing_utils
    MP_CONNECTIONS = MP_HANDS.HAND_CONNECTIONS
else:
    MP_HANDS = None
    MP_DRAW = None
    MP_CONNECTIONS = None

DEFAULT_YOLO_WEIGHTS = "yolo11n-pose.pt"

__all__ = ["HandGestureController", "DEFAULT_YOLO_WEIGHTS"]


class HandLandmark(IntEnum):
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


HAND_CONNECTIONS = [
    (HandLandmark.WRIST, HandLandmark.THUMB_CMC),
    (HandLandmark.THUMB_CMC, HandLandmark.THUMB_MCP),
    (HandLandmark.THUMB_MCP, HandLandmark.THUMB_IP),
    (HandLandmark.THUMB_IP, HandLandmark.THUMB_TIP),
    (HandLandmark.WRIST, HandLandmark.INDEX_MCP),
    (HandLandmark.INDEX_MCP, HandLandmark.INDEX_PIP),
    (HandLandmark.INDEX_PIP, HandLandmark.INDEX_DIP),
    (HandLandmark.INDEX_DIP, HandLandmark.INDEX_TIP),
    (HandLandmark.WRIST, HandLandmark.MIDDLE_MCP),
    (HandLandmark.MIDDLE_MCP, HandLandmark.MIDDLE_PIP),
    (HandLandmark.MIDDLE_PIP, HandLandmark.MIDDLE_DIP),
    (HandLandmark.MIDDLE_DIP, HandLandmark.MIDDLE_TIP),
    (HandLandmark.WRIST, HandLandmark.RING_MCP),
    (HandLandmark.RING_MCP, HandLandmark.RING_PIP),
    (HandLandmark.RING_PIP, HandLandmark.RING_DIP),
    (HandLandmark.RING_DIP, HandLandmark.RING_TIP),
    (HandLandmark.WRIST, HandLandmark.PINKY_MCP),
    (HandLandmark.PINKY_MCP, HandLandmark.PINKY_PIP),
    (HandLandmark.PINKY_PIP, HandLandmark.PINKY_DIP),
    (HandLandmark.PINKY_DIP, HandLandmark.PINKY_TIP),
]


def _landmark(hl, *names):
    """Return the first available landmark attr for compatibility across versions."""
    for name in names:
        if hasattr(hl, name):
            return getattr(hl, name)
    raise AttributeError(
        f"HandLandmark has none of the attributes {', '.join(names)}. "
        "Update the compatibility table."
    )


class HandGestureController:
    """Track a real hand with Ultralytics YOLO pose and emit AeroPiper physical control values."""

    def __init__(
        self,
        camera_index: int = 0,
        show_preview: bool = True,
        mirror_preview: bool = True,
        smoothing: float = 0.35,
        idle_decay: float = 0.97,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
        yolo_imgsz: int = 640,
        yolo_confidence: float = 0.45,
        yolo_device: Optional[str] = None,
        handedness: str = "left",
        pseudo_depth_scale: float = 0.08,
        max_hands: int = 1,
        backend: str = "yolo",
        mediapipe_fallback: bool = True,
        mediapipe_model_complexity: int = 1,
        mediapipe_detection_confidence: float = 0.6,
        mediapipe_tracking_confidence: float = 0.5,
        thumb_flexion_gain: float = 1.8,
        thumb_flexion_bias: float = 0.0,
        thumb_abd_gain: float = 1.5,
        thumb_abd_bias: float = 0.0,
        thumb_abd_invert: bool = False,
        thumb1_curve: float = 0.75,
        thumb2_curve: float = 1.05,
        thumb_abd_smoothing: float = 0.35,
        thumb_abd_deadband: float = 0.08,
        thumb1_freeze_seconds: float = 0.4,
        thumb1_disabled: bool = False,
    ):
        self._camera_index = camera_index
        self._show_preview = show_preview
        self._mirror_preview = mirror_preview
        self._alpha = float(np.clip(smoothing, 0.0, 1.0))
        self._idle_decay = float(np.clip(idle_decay, 0.80, 0.999))
        self._preview_window = "AeroPiper Left Hand"
        self._frame_provider = frame_provider
        self._owns_capture = frame_provider is None
        self._capture = self._open_camera(camera_index) if self._owns_capture else None
        self._physical = np.zeros(7, dtype=np.float32)
        self._initialized = False
        self._stop_requested = False
        self._last_detection_time = 0.0
        self._thumb_metric_blend = 0.8
        self._thumb_raw_min = None
        self._thumb_raw_max = None
        self._thumb_calibration_samples = 0
        self._last_landmarks: Optional[np.ndarray] = None
        backend_normalized = backend.lower()
        if backend_normalized not in ("yolo", "mediapipe"):
            raise ValueError("backend must be 'yolo' or 'mediapipe'")
        self._backend = backend_normalized
        self._last_backend_used: Optional[str] = None
        self._allow_mediapipe_fallback = bool(mediapipe_fallback)
        self._mp_tracker = None
        self._mp_model_complexity = int(np.clip(mediapipe_model_complexity, 0, 2))
        self._mp_detection_conf = float(np.clip(mediapipe_detection_confidence, 0.05, 1.0))
        self._mp_tracking_conf = float(np.clip(mediapipe_tracking_confidence, 0.05, 1.0))
        self._handedness = handedness.lower()
        if self._handedness not in ("left", "right"):
            raise ValueError("handedness must be 'left' or 'right'")
        self._pseudo_depth_scale = float(pseudo_depth_scale)
        self._yolo_imgsz = int(yolo_imgsz)
        self._yolo_confidence = float(np.clip(yolo_confidence, 0.05, 0.95))
        self._max_hands = max(1, int(max_hands))
        self._device = (
            yolo_device
            if yolo_device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._thumb_flexion_gain = float(max(thumb_flexion_gain, 0.1))
        self._thumb_flexion_bias = float(thumb_flexion_bias)
        self._thumb_abd_gain = float(max(thumb_abd_gain, 0.1))
        self._thumb_abd_bias = float(thumb_abd_bias)
        self._thumb_abd_invert = bool(thumb_abd_invert)
        self._thumb1_curve = float(max(min(thumb1_curve, 5.0), 0.1))
        self._thumb2_curve = float(max(min(thumb2_curve, 5.0), 0.1))
        self._thumb_abd_alpha = float(np.clip(thumb_abd_smoothing, 0.0, 1.0))
        self._thumb_abd_deadband = float(np.clip(thumb_abd_deadband, 0.0, 0.5))
        self._thumb_abd_filtered = None
        self._thumb1_freeze_duration = float(max(thumb1_freeze_seconds, 0.0))
        self._thumb1_freeze_until = (
            time.time() + self._thumb1_freeze_duration if self._thumb1_freeze_duration > 0 else 0.0
        )
        self._thumb1_hold_value = None
        self._warned_insufficient_keypoints = False
        self._model = None
        if self._backend == "yolo":
            self._model = self._load_model(yolo_weights)
        if self._backend == "mediapipe" or self._allow_mediapipe_fallback:
            self._maybe_init_mediapipe(required=self._backend == "mediapipe")
        self._physical_labels = ["TAbd", "Th1", "Th2", "Idx", "Mid", "Rng", "Pny"]
        self._thumb1_disabled = bool(thumb1_disabled)

    def _load_model(self, weights: str):
        resolved = self._resolve_weights(weights)
        try:
            model = YOLO(resolved)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Could not find YOLO weights at '{resolved}'. "
                "Either pass a valid local checkpoint via --yolo-weights or use a built-in "
                f"Ultralytics alias such as '{DEFAULT_YOLO_WEIGHTS}' (auto-downloaded on first run)."
            ) from exc
        if hasattr(model, "to"):
            model.to(self._device)
        return model

    def _resolve_weights(self, weights: str) -> str:
        if os.path.isabs(weights) and os.path.exists(weights):
            return weights
        local_path = os.path.join(os.path.dirname(__file__), "weights", weights)
        if os.path.exists(local_path):
            return local_path
        return weights

    def _maybe_init_mediapipe(self, required: bool = False):
        if self._mp_tracker is not None:
            return
        if mp is None:
            if required:
                raise ImportError(
                    "MediaPipe is required for the requested hand-tracking backend. "
                    "Install it via `pip install mediapipe`."
                )
            return
        self._mp_tracker = MP_HANDS.Hands(
            static_image_mode=False,
            model_complexity=self._mp_model_complexity,
            max_num_hands=1,
            min_detection_confidence=self._mp_detection_conf,
            min_tracking_confidence=self._mp_tracking_conf,
        )

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
        """Read the current frame, update gesture estimates, and return 7 physical DOFs."""
        if self._stop_requested:
            return self._physical.copy()

        frame = self._read_frame()
        if frame is None:
            self._stop_requested = True
            return self._physical.copy()

        measurement, annotated = self._extract_measurement(frame)
        now = time.time()

        if measurement is not None:
            if not self._initialized:
                self._physical = measurement
                self._initialized = True
            else:
                self._physical = (
                    self._alpha * measurement + (1.0 - self._alpha) * self._physical
                )
            self._last_detection_time = now
        elif self._initialized and now - self._last_detection_time > 0.2:
            self._physical *= self._idle_decay

        if self._show_preview:
            display = annotated if annotated is not None else frame
            self._render_preview(display)

        return self._physical.copy()

    def should_stop(self) -> bool:
        return self._stop_requested

    def reset_thumb_calibration(self):
        """Reset thumb abduction calibration bounds to recalibrate."""
        self._thumb_raw_min = None
        self._thumb_raw_max = None
        self._thumb_calibration_samples = 0
        self._thumb_zero_trigger = 0.35
        if self._thumb1_freeze_duration > 0.0:
            self._thumb1_freeze_until = time.time() + self._thumb1_freeze_duration
            self._thumb1_hold_value = None
        print("\n[CALIBRATION RESET] Move thumb NOW:")
        print("  1. Fully OPEN/abducted (spread wide)")
        print("  2. Fully CLOSED (against palm)")
        print("  3. Repeat for 3-4 seconds\n")

    def close(self):
        if self._mp_tracker is not None:
            if hasattr(self._mp_tracker, "close"):
                self._mp_tracker.close()
            self._mp_tracker = None
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
            if frame is None:
                return None
            return frame
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def _extract_measurement(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._backend == "yolo":
            physical, annotated = self._extract_measurement_yolo(frame)
            if physical is not None:
                return physical, annotated
            if self._allow_mediapipe_fallback and self._mp_tracker is not None:
                return self._extract_measurement_mediapipe(frame)
            return physical, annotated
        return self._extract_measurement_mediapipe(frame)

    def _extract_measurement_yolo(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._model is None:
            return None, None
        results = self._model.predict(
            source=frame,
            conf=self._yolo_confidence,
            imgsz=self._yolo_imgsz,
            device=self._device,
            verbose=False,
            max_det=self._max_hands,
        )
        if not results:
            self._last_landmarks = None
            return None, None

        detection = self._select_hand(results[0], frame.shape)
        if detection is None:
            self._last_landmarks = None
            return None, None

        normalized, pixel_points = detection
        physical = self._landmarks_to_physical(normalized)
        annotated = frame.copy()
        self._draw_landmarks(annotated, pixel_points)
        self._last_landmarks = pixel_points
        self._last_backend_used = "yolo"
        return physical, annotated

    def _extract_measurement_mediapipe(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        self._maybe_init_mediapipe(required=self._backend == "mediapipe")
        if self._mp_tracker is None:
            return None, None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_tracker.process(rgb)
        if not results.multi_hand_landmarks:
            self._last_landmarks = None
            return None, None

        selected = self._select_left_hand_mediapipe(results)
        if selected is None:
            self._last_landmarks = None
            return None, None

        points = np.array(
            [(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in selected.landmark],
            dtype=np.float32,
        )
        frame_h, frame_w = frame.shape[:2]
        pixel_points = np.zeros((points.shape[0], 2), dtype=np.float32)
        pixel_points[:, 0] = np.clip(points[:, 0], 0.0, 1.0) * frame_w
        pixel_points[:, 1] = np.clip(points[:, 1], 0.0, 1.0) * frame_h
        annotated = frame.copy()
        self._draw_landmarks(annotated, pixel_points)
        self._last_landmarks = pixel_points
        self._last_backend_used = "mediapipe"
        physical = self._landmarks_to_physical(points)
        return physical, annotated

    @staticmethod
    def _select_left_hand_mediapipe(results):
        handedness_list = getattr(results, "multi_handedness", None)
        if handedness_list:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, handedness_list
            ):
                label = handedness.classification[0].label.lower()
                if label == "left":
                    return hand_landmarks
        return results.multi_hand_landmarks[0]

    def _select_hand(self, result, frame_shape):
        keypoints = getattr(result, "keypoints", None)
        boxes = getattr(result, "boxes", None)
        if keypoints is None or keypoints.data is None or keypoints.data.shape[0] == 0:
            return None

        keypoints_xy = keypoints.xy.cpu().numpy()
        if keypoints_xy.shape[1] < len(HandLandmark):
            self._warn_yolo_keypoint_count(keypoints_xy.shape[1])
            return None

        keypoints_conf = (
            keypoints.conf.cpu().numpy() if getattr(keypoints, "conf", None) is not None else None
        )
        boxes_xywh = boxes.xywh.cpu().numpy() if boxes is not None else None

        best_idx = None
        best_score = -np.inf
        for idx in range(keypoints_xy.shape[0]):
            sample = keypoints_xy[idx]
            kp_conf = keypoints_conf[idx] if keypoints_conf is not None else None
            box = boxes_xywh[idx] if boxes_xywh is not None and idx < boxes_xywh.shape[0] else None
            score = self._score_detection(sample, box, kp_conf)
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            return None

        sample = keypoints_xy[best_idx]
        box = boxes_xywh[best_idx] if boxes_xywh is not None and best_idx < boxes_xywh.shape[0] else None
        frame_h, frame_w = frame_shape[:2]

        normalized = np.zeros((sample.shape[0], 3), dtype=np.float32)
        normalized[:, 0] = sample[:, 0] / max(frame_w, 1e-2)
        normalized[:, 1] = sample[:, 1] / max(frame_h, 1e-2)
        normalized[:, 2] = self._estimate_depth(sample, box)

        return normalized, sample.astype(np.float32)

    def _score_detection(self, sample, box, kp_conf):
        thumb_tip = sample[HandLandmark.THUMB_TIP.value]
        pinky_tip = sample[HandLandmark.PINKY_TIP.value]
        width = box[2] if box is not None else np.linalg.norm(thumb_tip - pinky_tip)
        width = max(width, 1e-3)
        orientation = (thumb_tip[0] - pinky_tip[0]) / width
        if self._handedness == "right":
            orientation = -orientation
        area_score = 0.0
        if box is not None:
            area_score = np.log(max(box[2] * box[3], 1.0))
        conf_score = 0.0
        if kp_conf is not None:
            conf_score = float(np.mean(kp_conf))
        return 0.6 * orientation + 0.3 * area_score + 0.1 * conf_score

    def _estimate_depth(self, sample, box):
        if box is None:
            return np.zeros(sample.shape[0], dtype=np.float32)
        center_y = box[1]
        scale = max(box[3], 1e-3)
        depth = (sample[:, 1] - center_y) / scale
        depth *= self._pseudo_depth_scale
        return depth.astype(np.float32)

    def _warn_yolo_keypoint_count(self, observed: int):
        if self._warned_insufficient_keypoints:
            return
        required = len(HandLandmark)
        print(
            f"[WARN] YOLO pose weights output {observed} keypoints per hand, "
            f"but {required} are required for finger articulation."
        )
        if self._allow_mediapipe_fallback and self._mp_tracker is not None:
            print(
                "Falling back to MediaPipe Hand Landmarks until a hand-specific YOLO "
                "checkpoint (21 keypoints) is provided via --yolo-weights."
            )
        else:
            print(
                "Provide a 21-keypoint hand pose checkpoint via --yolo-weights or run "
                "the script with --hand-backend mediapipe."
            )
        self._warned_insufficient_keypoints = True

    def _landmarks_to_physical(self, hand_landmarks) -> np.ndarray:
        if isinstance(hand_landmarks, np.ndarray):
            points = hand_landmarks
        elif hasattr(hand_landmarks, "landmark"):
            points = np.array(
                [(lm.x, lm.y, getattr(lm, "z", 0.0)) for lm in hand_landmarks.landmark],
                dtype=np.float32,
            )
        else:
            raise TypeError("Unsupported hand_landmarks format")

        if points.shape[0] < len(HandLandmark):
            raise ValueError(
                f"Expected at least {len(HandLandmark)} points, received {points.shape[0]}"
            )

        hl = HandLandmark

        values = np.zeros(7, dtype=np.float32)
        values[0] = self._thumb_abduction(points, hl)
        values[1] = self._apply_thumb_flexion_adjust(
            self._joint_flexion(
                points,
                hl.THUMB_CMC,
                hl.THUMB_MCP,
                hl.THUMB_IP,
                open_deg=5.0,
                closed_deg=65.0,
            )
        )
        values[1] = self._apply_thumb1_curve(values[1])
        values[2] = self._apply_thumb_flexion_adjust(
            self._joint_flexion(
                points,
                hl.THUMB_MCP,
                hl.THUMB_IP,
                hl.THUMB_TIP,
                open_deg=5.0,
                closed_deg=115.0,
            )
        )
        values[2] = self._apply_thumb2_curve(values[2])
        values[3] = self._finger_curl(
            points,
            _landmark(hl, "INDEX_MCP", "INDEX_FINGER_MCP"),
            _landmark(hl, "INDEX_PIP", "INDEX_FINGER_PIP"),
            _landmark(hl, "INDEX_TIP", "INDEX_FINGER_TIP"),
            open_deg=5.0,
            closed_deg=110.0,
        )
        values[4] = self._finger_curl(
            points,
            _landmark(hl, "MIDDLE_MCP", "MIDDLE_FINGER_MCP"),
            _landmark(hl, "MIDDLE_PIP", "MIDDLE_FINGER_PIP"),
            _landmark(hl, "MIDDLE_TIP", "MIDDLE_FINGER_TIP"),
            open_deg=5.0,
            closed_deg=110.0,
        )
        values[5] = self._finger_curl(
            points,
            _landmark(hl, "RING_MCP", "RING_FINGER_MCP"),
            _landmark(hl, "RING_PIP", "RING_FINGER_PIP"),
            _landmark(hl, "RING_TIP", "RING_FINGER_TIP"),
            open_deg=5.0,
            closed_deg=110.0,
        )
        values[6] = self._finger_curl(
            points,
            _landmark(hl, "PINKY_MCP", "PINKY_FINGER_MCP"),
            _landmark(hl, "PINKY_PIP", "PINKY_FINGER_PIP"),
            _landmark(hl, "PINKY_TIP", "PINKY_FINGER_TIP"),
            open_deg=5.0,
            closed_deg=110.0,
        )

        return np.clip(values * 100.0, 0.0, 100.0)

    def _apply_thumb_flexion_adjust(self, normalized_val: float) -> float:
        adjusted = normalized_val * self._thumb_flexion_gain + self._thumb_flexion_bias
        return float(np.clip(adjusted, 0.0, 1.0))

    def _apply_thumb1_curve(self, normalized_val: float) -> float:
        curve = self._thumb1_curve
        if abs(curve - 1.0) < 1e-3:
            return normalized_val
        curved = normalized_val ** curve
        return float(np.clip(curved, 0.0, 1.0))

    def _apply_thumb2_curve(self, normalized_val: float) -> float:
        curve = self._thumb2_curve
        if abs(curve - 1.0) < 1e-3:
            return normalized_val
        curved = normalized_val ** curve
        return float(np.clip(curved, 0.0, 1.0))

    def _apply_thumb1_freeze(self, value: float) -> float:
        if self._thumb1_freeze_duration <= 0.0:
            self._thumb1_hold_value = value
            return float(value)
        now = time.time()
        if now < self._thumb1_freeze_until:
            if self._thumb1_hold_value is None:
                self._thumb1_hold_value = value
            return float(self._thumb1_hold_value)
        self._thumb1_hold_value = value
        return float(value)

    def _finger_curl(self, points, mcp, pip, tip, open_deg: float, closed_deg: float):
        base = points[mcp.value]
        mid = points[pip.value]
        fingertip = points[tip.value]
        v1 = mid - base
        v2 = fingertip - mid
        return self._angle_to_ratio(v1, v2, open_deg, closed_deg)

    def _joint_flexion(
        self,
        points,
        root_idx,
        mid_idx,
        tip_idx,
        open_deg: float,
        closed_deg: float,
    ):
        root = points[root_idx.value]
        mid = points[mid_idx.value]
        tip = points[tip_idx.value]
        v1 = mid - root
        v2 = tip - mid
        return self._angle_to_ratio(v1, v2, open_deg, closed_deg)

    def _thumb_abduction(self, points, hl):
        wrist = points[hl.WRIST.value]
        thumb_mcp = points[hl.THUMB_MCP.value]
        index_landmark = _landmark(hl, "INDEX_MCP", "INDEX_FINGER_MCP")
        index_mcp = points[index_landmark.value]

        thumb_vec = thumb_mcp - wrist
        index_vec = index_mcp - wrist

        angle_ratio = self._angle_to_ratio(thumb_vec, index_vec, 0.0, 90.0)

        thumb_tip = points[hl.THUMB_TIP.value]
        gap = np.linalg.norm(thumb_tip - index_mcp)

        pinky_mcp = points[_landmark(hl, "PINKY_MCP", "PINKY_FINGER_MCP").value]
        palm_width = np.linalg.norm(index_mcp - pinky_mcp)
        if palm_width < 1e-4:
            palm_width = 1e-4
        gap_normalized = gap / palm_width

        gap_ratio = np.clip((gap_normalized - 0.0) / (1.5 - 0.0), 0.0, 1.0)

        combined = (
            self._thumb_metric_blend * angle_ratio
            + (1.0 - self._thumb_metric_blend) * gap_ratio
        )

        raw_val = 1.0 - combined
        raw_val = self._thumb_zero_compensate(raw_val)

        # 4. Hardcoded dynamic calibration: track actual min/max over first 100 samples
        if self._thumb_calibration_samples < 100:
            if self._thumb_raw_min is None:
                self._thumb_raw_min = raw_val
                self._thumb_raw_max = raw_val
            else:
                self._thumb_raw_min = min(self._thumb_raw_min, raw_val)
                self._thumb_raw_max = max(self._thumb_raw_max, raw_val)
            self._thumb_calibration_samples += 1

        if self._thumb_raw_min is not None and self._thumb_raw_max is not None:
            range_span = self._thumb_raw_max - self._thumb_raw_min
            padded_min = self._thumb_raw_min - range_span * 0.3
            padded_max = self._thumb_raw_max + range_span * 0.15
            if padded_max - padded_min > 0.05:
                val = (raw_val - padded_min) / (padded_max - padded_min)
                return self._smooth_thumb_abd(self._apply_thumb_abd_adjust(val))

        return self._smooth_thumb_abd(
            self._apply_thumb_abd_adjust(float(np.clip(raw_val, 0.0, 1.0)))
        )

    @staticmethod
    def _angle_to_ratio(v1, v2, open_deg: float, closed_deg: float):
        angle = HandGestureController._vector_angle(v1, v2)
        open_rad = np.deg2rad(open_deg)
        closed_rad = np.deg2rad(closed_deg)
        if not np.isfinite(open_rad):
            open_rad = 0.0
        if not np.isfinite(closed_rad) or closed_rad <= open_rad + 1e-4:
            closed_rad = open_rad + 1.0
        span = closed_rad - open_rad
        normalized = (angle - open_rad) / span
        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _vector_angle(v1, v2):
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        if v1_norm < 1e-5 or v2_norm < 1e-5:
            return 0.0
        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return float(np.arccos(cos_angle))

    def _apply_thumb_abd_adjust(self, normalized_val: float) -> float:
        adjusted = normalized_val * self._thumb_abd_gain + self._thumb_abd_bias
        if self._thumb_abd_invert:
            adjusted = 1.0 - adjusted
        return float(np.clip(adjusted, 0.0, 1.0))

    def _smooth_thumb_abd(self, value: float) -> float:
        alpha = self._thumb_abd_alpha
        if alpha <= 0.0:
            self._thumb_abd_filtered = value
            return float(value)
        if self._thumb_abd_filtered is None:
            filtered = value
        else:
            filtered = alpha * value + (1.0 - alpha) * self._thumb_abd_filtered
        if filtered < self._thumb_abd_deadband:
            filtered = 0.0
        self._thumb_abd_filtered = filtered
        return float(filtered)

    def _thumb_zero_compensate(self, value: float) -> float:
        """Simple passthrough placeholder to maintain compatibility."""
        return float(np.clip(value, 0.0, 1.0))

    def _render_preview(self, frame):
        if frame is None or not self._show_preview:
            return
        overlay = frame.copy()
        self.annotate_frame(overlay)
        if self._mirror_preview:
            overlay = cv2.flip(overlay, 1)
        backend_label = (self._last_backend_used or self._backend).upper()
        cv2.putText(
            overlay,
            f"{backend_label} hand tracking (Q/ESC exit, R reset thumb cal)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        stats = " ".join(
            f"{name}:{val:5.1f}"
            for name, val in zip(self._physical_labels, self._physical)
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
            elif key == ord("r"):
                self.reset_thumb_calibration()
        except cv2.error:
            print("OpenCV preview unavailable, continuing without it.")
            self._show_preview = False

    def annotate_frame(self, frame: Optional[np.ndarray]):
        if frame is None or self._last_landmarks is None:
            return
        self._draw_landmarks(frame, self._last_landmarks)

    def _draw_landmarks(self, frame: np.ndarray, points: np.ndarray):
        for start, end in HAND_CONNECTIONS:
            pt1 = points[start.value]
            pt2 = points[end.value]
            if not np.all(np.isfinite(pt1)) or not np.all(np.isfinite(pt2)):
                continue
            cv2.line(
                frame,
                (int(pt1[0]), int(pt1[1])),
                (int(pt2[0]), int(pt2[1])),
                (0, 255, 0),
                2,
                lineType=cv2.LINE_AA,
            )
        for idx, pt in enumerate(points):
            if not np.all(np.isfinite(pt)):
                continue
            radius = 4 if idx in (HandLandmark.WRIST,) else 3
            cv2.circle(
                frame,
                (int(pt[0]), int(pt[1])),
                radius,
                (0, 128, 255) if idx < 5 else (255, 0, 0),
                -1,
                lineType=cv2.LINE_AA,
            )


