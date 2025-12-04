"""
Single-camera controller that fuses arm + hand tracking into one interface.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import cv2  # type: ignore[import]
import numpy as np

from arm_tracker import ArmGestureController  # type: ignore[import]
from hand_tracker import HandGestureController  # type: ignore[import]

ARM_LABELS = ["BaseYaw", "ShoulderPitch", "ElbowFlex", "WristRoll"]
HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]


class ComboGestureController:
    """
    Wraps the arm + hand trackers so they share one webcam and preview window.
    """

    def __init__(
        self,
        camera_index: int = 0,
        show_preview: bool = True,
        mirror_preview: bool = True,
        arm_smoothing: float = 0.5,
        arm_idle_decay: float = 0.96,
        arm_visibility_threshold: float = 0.45,
        max_step: float = 0.35,
        hand_smoothing: float = 0.35,
        hand_idle_decay: float = 0.97,
    ):
        self._camera_index = camera_index
        self._show_preview = show_preview
        self._preview_window = "AeroPiper Arm + Hand"
        self._mirror_preview = mirror_preview
        self._stop_requested = False
        self._frame: Optional[np.ndarray] = None

        self._capture = self._open_camera(camera_index)

        def _frame_provider():
            return self._frame

        self._arm_controller = ArmGestureController(
            camera_index=camera_index,
            show_preview=False,
            mirror_preview=False,
            smoothing=arm_smoothing,
            idle_decay=arm_idle_decay,
            visibility_threshold=arm_visibility_threshold,
            max_step=max_step,
            frame_provider=_frame_provider,
        )
        self._hand_controller = HandGestureController(
            camera_index=camera_index,
            show_preview=False,
            mirror_preview=False,
            smoothing=hand_smoothing,
            idle_decay=hand_idle_decay,
            frame_provider=_frame_provider,
        )

        self._arm_values = np.zeros(len(ARM_LABELS), dtype=np.float32)
        self._hand_values = np.zeros(len(HAND_LABELS), dtype=np.float32)
        self._executor = ThreadPoolExecutor(max_workers=2)

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

    def _read_frame(self) -> Optional[np.ndarray]:
        if self._capture is None:
            return None
        ok, frame = self._capture.read()
        if not ok:
            return None
        return frame

    def update(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._stop_requested:
            return self._arm_values.copy(), self._hand_values.copy()

        frame = self._read_frame()
        if frame is None:
            self._stop_requested = True
            return self._arm_values.copy(), self._hand_values.copy()

        self._frame = frame
        arm_future = self._executor.submit(self._arm_controller.update)
        hand_future = self._executor.submit(self._hand_controller.update)
        arm = arm_future.result()
        hand = hand_future.result()
        self._arm_values = arm
        self._hand_values = hand

        if self._show_preview:
            self._render_preview(frame)

        return arm.copy(), hand.copy()

    def should_stop(self) -> bool:
        return (
            self._stop_requested
            or self._arm_controller.should_stop()
            or self._hand_controller.should_stop()
        )

    def close(self):
        self._arm_controller.close()
        self._hand_controller.close()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        if self._show_preview:
            try:
                cv2.destroyWindow(self._preview_window)
            except cv2.error:
                pass

    def _render_preview(self, frame: np.ndarray):
        if frame is None or not self._show_preview:
            return

        overlay = frame.copy()
        self._arm_controller.annotate_frame(overlay)
        self._hand_controller.annotate_frame(overlay)
        if self._mirror_preview:
            overlay = cv2.flip(overlay, 1)
        arm_stats = " ".join(
            f"{name}:{val:+4.2f}" for name, val in zip(ARM_LABELS, self._arm_values)
        )
        hand_stats = " ".join(
            f"{name}:{val:5.1f}" for name, val in zip(HAND_LABELS, self._hand_values)
        )
        cv2.putText(
            overlay,
            "Left arm + hand tracking (Q/ESC exit, R reset thumb cal)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            arm_stats,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            hand_stats,
            (10, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        try:
            cv2.imshow(self._preview_window, overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._stop_requested = True
            elif key == ord("r"):
                self._hand_controller.reset_thumb_calibration()
                print("[Combo] Thumb abduction calibration reset.")
        except cv2.error:
            print("OpenCV preview unavailable, continuing without it.")
            self._show_preview = False

