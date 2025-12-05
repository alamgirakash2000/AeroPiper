"""
Reusable camera preview helper for MuJoCo viewers.

- Tries to import OpenCV; if unavailable, update() is a no-op and logs once.
- Call update(viewer) inside your sim loop; call close() on exit.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None  # type: ignore


class CameraPreviewer:
    def __init__(
        self,
        camera_names: Iterable[str],
        interval: float = 0.1,
        log_prefix: str = "camera_preview",
    ) -> None:
        self.camera_names = list(camera_names)
        self.interval = interval
        self.log_prefix = log_prefix
        self._last_update = 0.0
        self._warned = False
        self._announced_active = False
        self._opened_window = set()
        self._missing_cam_log = set()

    def update(self, viewer) -> None:
        """Render cameras to OpenCV windows if cv2 is available."""
        if not _HAS_CV2:
            if not self._warned:
                # Log once to avoid spamming the console
                print(f"[{self.log_prefix}] opencv-python not installed; camera preview disabled.")
                self._warned = True
            return

        if not self._announced_active:
            print(f"[{self.log_prefix}] camera preview active via OpenCV. Close the cv2 windows to hide.")
            self._announced_active = True

        now = time.time()
        if (now - self._last_update) < self.interval:
            return

        for cam_name in self.camera_names:
            try:
                rgb, _ = viewer.read_pixels(camera_name=cam_name)
            except Exception:
                if cam_name not in self._missing_cam_log:
                    print(f"[{self.log_prefix}] could not read camera '{cam_name}' (name/id missing?)")
                    self._missing_cam_log.add(cam_name)
                continue
            if rgb is None:
                continue
            if cam_name not in self._opened_window:
                cv2.namedWindow(cam_name, cv2.WINDOW_AUTOSIZE)
                print(f"[{self.log_prefix}] showing feed for '{cam_name}'")
                self._opened_window.add(cam_name)
            # Convert RGB to BGR for OpenCV
            bgr = rgb[:, :, ::-1].copy()
            cv2.imshow(cam_name, bgr)

        cv2.waitKey(1)
        self._last_update = now

    def close(self) -> None:
        if _HAS_CV2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
