import os
import time
from typing import Optional, Tuple

try:
    import cv2  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "OpenCV (cv2) is required for webcam hand tracking. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install opencv-python`."
    ) from exc

try:
    import mediapipe as mp  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "MediaPipe is required for real-time hand landmark tracking. "
        "Activate the 'aeropiper' conda environment and install it via "
        "`conda activate aeropiper && pip install mediapipe`."
    ) from exc

import numpy as np

MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils
MP_CONNECTIONS = MP_HANDS.HAND_CONNECTIONS

__all__ = ["HandGestureController"]


def _landmark(hl, *names):
    """Return the first available landmark attr for compatibility across versions."""
    for name in names:
        if hasattr(hl, name):
            return getattr(hl, name)
    raise AttributeError(
        f"HandLandmark has none of the attributes {', '.join(names)}. "
        "Upgrade/downgrade MediaPipe or update the compatibility table."
    )


class HandGestureController:
    """Track a real left hand and emit AeroPiper physical control values."""

    _THUMB_ABD_MIN = np.deg2rad(20.0)
    _THUMB_ABD_MAX = np.deg2rad(50.0)

    def __init__(
        self,
        camera_index: int = 0,
        show_preview: bool = True,
        smoothing: float = 0.35,
        idle_decay: float = 0.97,
    ):
        self._camera_index = camera_index
        self._show_preview = show_preview
        self._alpha = float(np.clip(smoothing, 0.0, 1.0))
        self._idle_decay = float(np.clip(idle_decay, 0.80, 0.999))
        self._preview_window = "AeroPiper Left Hand"
        self._capture = self._open_camera(camera_index)
        self._hands = MP_HANDS.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._physical = np.zeros(7, dtype=np.float32)
        self._initialized = False
        self._stop_requested = False
        self._last_detection_time = 0.0
        self._thumb_metric_blend = 0.8
        
        # Hardcoded thumb abduction calibration tracking
        self._thumb_raw_min = None
        self._thumb_raw_max = None
        self._thumb_calibration_samples = 0
        
        print("\n[Thumb Calibration] Starting in 1 second...")
        print("  Move your thumb through FULL range:")
        print("  - Fully OPEN/abducted (spread wide from palm)")
        print("  - Fully CLOSED (pressed against palm)")
        print("  Repeat for 3-4 seconds for best calibration\n")

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
        print("\n[CALIBRATION RESET] Move thumb NOW:")
        print("  1. Fully OPEN/abducted (spread wide)")
        print("  2. Fully CLOSED (against palm)")
        print("  3. Repeat for 3-4 seconds\n")

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if hasattr(self._hands, "close"):
            self._hands.close()
        if self._show_preview:
            try:
                cv2.destroyWindow(self._preview_window)
            except cv2.error:
                pass

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def _extract_measurement(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return None, None

        selected = self._select_left_hand(results)
        if selected is None:
            return None, None

        physical = self._landmarks_to_physical(selected)
        annotated = frame.copy()
        MP_DRAW.draw_landmarks(annotated, selected, MP_CONNECTIONS)
        return physical, annotated

    @staticmethod
    def _select_left_hand(results):
        handedness_list = getattr(results, "multi_handedness", None)
        if handedness_list:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, handedness_list
            ):
                label = handedness.classification[0].label.lower()
                if label == "left":
                    return hand_landmarks
        return results.multi_hand_landmarks[0]

    def _landmarks_to_physical(self, hand_landmarks) -> np.ndarray:
        points = np.array(
            [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32
        )
        hl = MP_HANDS.HandLandmark

        values = np.zeros(7, dtype=np.float32)
        values[0] = self._thumb_abduction(points, hl)
        values[1] = self._joint_flexion(
            points,
            hl.THUMB_CMC,
            hl.THUMB_MCP,
            hl.THUMB_IP,
            open_deg=5.0,
            closed_deg=65.0,
        )
        values[2] = self._joint_flexion(
            points,
            hl.THUMB_MCP,
            hl.THUMB_IP,
            hl.THUMB_TIP,
            open_deg=5.0,
            closed_deg=115.0,
        )
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
        # 1. Calculate the angle between thumb and index metacarpals (in degrees)
        wrist = points[hl.WRIST.value]
        thumb_mcp = points[hl.THUMB_MCP.value]
        index_landmark = _landmark(hl, "INDEX_MCP", "INDEX_FINGER_MCP")
        index_mcp = points[index_landmark.value]
        
        thumb_vec = thumb_mcp - wrist
        index_vec = index_mcp - wrist
        
        # Use extremely wide angle bounds to capture full range: 0 deg (closed) to 90 deg (wide open)
        angle_ratio = self._angle_to_ratio(thumb_vec, index_vec, 0.0, 90.0)

        # 2. Calculate gap normalized by palm width to handle hand size differences
        thumb_tip = points[hl.THUMB_TIP.value]
        gap = np.linalg.norm(thumb_tip - index_mcp)
        
        pinky_mcp = points[_landmark(hl, "PINKY_MCP", "PINKY_FINGER_MCP").value]
        palm_width = np.linalg.norm(index_mcp - pinky_mcp)
        
        # Safety against zero division
        if palm_width < 1e-4:
            palm_width = 1e-4
            
        gap_normalized = gap / palm_width
        
        # Map normalized gap with extremely wide range: 0.0 (closed) to 1.5 (wide open)
        gap_ratio = np.clip((gap_normalized - 0.0) / (1.5 - 0.0), 0.0, 1.0)

        # 3. Combine metrics
        # Blend mostly on angle (robust) but use gap for fine control
        combined = (
            self._thumb_metric_blend * angle_ratio
            + (1.0 - self._thumb_metric_blend) * gap_ratio
        )
        
        # Invert so that Open (Large Angle/Gap) = 0 and Closed = 100
        raw_val = 1.0 - combined
        
        # 4. Hardcoded dynamic calibration: track actual min/max over first 100 samples
        if self._thumb_calibration_samples < 100:
            if self._thumb_raw_min is None:
                self._thumb_raw_min = raw_val
                self._thumb_raw_max = raw_val
            else:
                self._thumb_raw_min = min(self._thumb_raw_min, raw_val)
                self._thumb_raw_max = max(self._thumb_raw_max, raw_val)
            self._thumb_calibration_samples += 1
            
            # Print calibration progress
            if self._thumb_calibration_samples == 100:
                print(f"[ThumbAbd Calibration] Min={self._thumb_raw_min:.3f}, Max={self._thumb_raw_max:.3f}")
        
        # Remap using calibrated range with aggressive padding on minimum side
        if self._thumb_raw_min is not None and self._thumb_raw_max is not None:
            # Add more aggressive padding to the range, especially on minimum
            range_span = self._thumb_raw_max - self._thumb_raw_min
            padded_min = self._thumb_raw_min - range_span * 0.3  # 30% padding on min side
            padded_max = self._thumb_raw_max + range_span * 0.15  # 15% padding on max side
            
            if padded_max - padded_min > 0.05:  # Ensure reasonable range
                val = (raw_val - padded_min) / (padded_max - padded_min)
                return float(np.clip(val, 0.0, 1.0))
        
        # Fallback: use raw value
        return float(np.clip(raw_val, 0.0, 1.0))

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

    def _render_preview(self, frame):
        if frame is None or not self._show_preview:
            return
        overlay = frame.copy()
        cv2.putText(
            overlay,
            "Left hand tracking (Q/ESC exit, R reset thumb cal)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        labels = ["TAbd", "Th1", "Th2", "Idx", "Mid", "Rng", "Pny"]
        stats = " ".join(
            f"{name}:{val:5.1f}" for name, val in zip(labels, self._physical)
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
                print("Thumb abduction calibration reset.")
        except cv2.error:
            print("OpenCV preview unavailable, continuing without it.")
            self._show_preview = False


