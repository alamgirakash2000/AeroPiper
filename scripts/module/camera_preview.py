"""
Reusable camera preview helper for MuJoCo viewers.

- Tries to import OpenCV; if unavailable, update() is a no-op and logs once.
- Call update(data) inside your sim loop; call close() on exit.
"""

from __future__ import annotations

import time
from typing import Iterable, Optional

import mujoco

try:
    import cv2  # type: ignore

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False
    cv2 = None  # type: ignore


class CameraPreviewer:
    def __init__(
        self,
        model: mujoco.MjModel,
        camera_names: Iterable[str],
        interval: float = 0.1,
        log_prefix: str = "camera_preview",
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.model = model
        self.camera_names = list(camera_names)
        self.interval = interval
        self.log_prefix = log_prefix
        self.width = width
        self.height = height
        self._last_update = 0.0
        self._warned = False
        self._announced_active = False
        self._opened_window = set()
        self._missing_cam_log = set()
        self._renderer = None
        self._camera_ids = {}
        
        # Resolve camera IDs at init
        for cam_name in self.camera_names:
            try:
                cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                if cam_id >= 0:
                    self._camera_ids[cam_name] = cam_id
                else:
                    print(f"[{self.log_prefix}] Warning: camera '{cam_name}' not found in model")
            except Exception as e:
                print(f"[{self.log_prefix}] Error resolving camera '{cam_name}': {e}")
        
        # Create renderer if we have OpenCV and valid cameras
        if _HAS_CV2 and self._camera_ids:
            try:
                self._renderer = mujoco.Renderer(self.model, self.height, self.width)
            except Exception as e:
                print(f"[{self.log_prefix}] Failed to create renderer: {e}")

    def update(self, data: mujoco.MjData) -> None:
        """Render cameras to OpenCV windows if cv2 is available."""
        if not _HAS_CV2:
            if not self._warned:
                print(f"[{self.log_prefix}] opencv-python not installed; camera preview disabled.")
                self._warned = True
            return

        if self._renderer is None:
            if not self._warned:
                print(f"[{self.log_prefix}] No renderer available; camera preview disabled.")
                self._warned = True
            return

        if not self._announced_active:
            print(f"[{self.log_prefix}] camera preview active via OpenCV. Close the cv2 windows to hide.")
            self._announced_active = True

        now = time.time()
        if (now - self._last_update) < self.interval:
            return

        for cam_name, cam_id in self._camera_ids.items():
            try:
                # Update renderer to use this camera
                self._renderer.update_scene(data, camera=cam_id)
                # Render and get pixels
                rgb = self._renderer.render()
                
                if rgb is not None:
                    if cam_name not in self._opened_window:
                        cv2.namedWindow(cam_name, cv2.WINDOW_AUTOSIZE)
                        print(f"[{self.log_prefix}] showing feed for '{cam_name}'")
                        self._opened_window.add(cam_name)
                    # Convert RGB to BGR for OpenCV and flip vertically
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    bgr = cv2.flip(bgr, 0)  # MuJoCo renders upside down
                    cv2.imshow(cam_name, bgr)
            except Exception as e:
                if cam_name not in self._missing_cam_log:
                    print(f"[{self.log_prefix}] Error rendering camera '{cam_name}': {e}")
                    self._missing_cam_log.add(cam_name)
                continue

        cv2.waitKey(1)
        self._last_update = now

    def close(self) -> None:
        if _HAS_CV2:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if self._renderer is not None:
            self._renderer.close()
