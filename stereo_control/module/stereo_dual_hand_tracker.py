"""
Stereo Dual Hand Tracker - Tracks BOTH hands in 3D

Tracks both left and right hands separately using stereo vision.
Returns values for both, but you can choose which to use for control.
"""

import time
import warnings
from typing import Optional, Tuple

try:
    import cv2
except ImportError as exc:
    raise ImportError("OpenCV required: pip install opencv-python") from exc

try:
    import mediapipe as mp
except ImportError as exc:
    raise ImportError("MediaPipe required: pip install mediapipe") from exc

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype")

MP_HANDS = mp.solutions.hands
MP_DRAW = mp.solutions.drawing_utils
MP_CONNECTIONS = MP_HANDS.HAND_CONNECTIONS


def _landmark(hl, *names):
    """Return first available landmark attribute."""
    for name in names:
        if hasattr(hl, name):
            return getattr(hl, name)
    raise AttributeError(f"HandLandmark has none of: {', '.join(names)}")


class StereoDualHandTracker:
    """Track BOTH hands in 3D using stereo cameras."""
    
    def __init__(
        self,
        calibration: dict,
        left_camera_index: int = 0,
        right_camera_index: int = 1,
        show_preview: bool = True,
        mirror_preview: bool = True,
        smoothing: float = 0.35,
    ):
        self.calibration = calibration
        self._show_preview = show_preview
        self._mirror_preview = mirror_preview
        self._alpha = float(np.clip(smoothing, 0.0, 1.0))
        
        # Open cameras
        self._cap_left = self._open_camera(left_camera_index)
        self._cap_right = self._open_camera(right_camera_index)
        
        # MediaPipe - detect BOTH hands - use LITE for speed
        self._hands_left = MP_HANDS.Hands(
            static_image_mode=False,
            model_complexity=0,  # LITE model = faster
            max_num_hands=2,  # Detect both
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands_right = MP_HANDS.Hands(
            static_image_mode=False,
            model_complexity=0,  # LITE model = faster
            max_num_hands=2,  # Detect both
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # State for BOTH hands
        self._left_hand_physical = np.zeros(7, dtype=np.float32)
        self._right_hand_physical = np.zeros(7, dtype=np.float32)
        self._left_initialized = False
        self._right_initialized = False
        self._stop_requested = False
        
        # Visualization
        self._last_left_hand_3d = None
        self._last_right_hand_3d = None
        self._preview_window = "Stereo Dual Hand Tracking"
    
    def _open_camera(self, index: int):
        """Open camera."""
        import os
        if os.name == "nt":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        
        width, height = self.calibration["image_size"]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        return cap
    
    def update(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Update tracking for BOTH hands.
        
        Returns:
            (left_hand_values [7], right_hand_values [7]) or (None, None) if not detected
            Values are in [0-100] scale
        """
        if self._stop_requested:
            return (
                self._left_hand_physical.copy() if self._left_initialized else None,
                self._right_hand_physical.copy() if self._right_initialized else None
            )
        
        # Read frames
        ret_l, frame_l = self._cap_left.read()
        ret_r, frame_r = self._cap_right.read()
        
        if not ret_l or not ret_r:
            self._stop_requested = True
            return None, None
        
        # Rectify (use NEAREST for speed!)
        frame_l_rect = cv2.remap(
            frame_l,
            self.calibration["map_left"][0],
            self.calibration["map_left"][1],
            cv2.INTER_NEAREST  # Faster than LINEAR
        )
        frame_r_rect = cv2.remap(
            frame_r,
            self.calibration["map_right"][0],
            self.calibration["map_right"][1],
            cv2.INTER_NEAREST  # Faster than LINEAR
        )
        
        # Track BOTH hands
        left_values, right_values, annotated_l, annotated_r = self._track_both_hands_3d(
            frame_l_rect, frame_r_rect
        )
        
        # Update left hand
        if left_values is not None:
            if not self._left_initialized:
                self._left_hand_physical = left_values
                self._left_initialized = True
            else:
                self._left_hand_physical = (
                    self._alpha * left_values + (1.0 - self._alpha) * self._left_hand_physical
                )
        
        # Update right hand
        if right_values is not None:
            if not self._right_initialized:
                self._right_hand_physical = right_values
                self._right_initialized = True
            else:
                self._right_hand_physical = (
                    self._alpha * right_values + (1.0 - self._alpha) * self._right_hand_physical
                )
        
        # Show preview
        if self._show_preview:
            display_l = annotated_l if annotated_l is not None else frame_l_rect
            display_r = annotated_r if annotated_r is not None else frame_r_rect
            self._render_preview(display_l, display_r)
        
        return (
            self._left_hand_physical.copy() if self._left_initialized else None,
            self._right_hand_physical.copy() if self._right_initialized else None
        )
    
    def _track_both_hands_3d(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Track both hands in 3D."""
        rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        
        results_l = self._hands_left.process(rgb_l)
        results_r = self._hands_right.process(rgb_r)
        
        # Extract hands with labels from both cameras
        hands_cam_l = self._extract_hands_with_labels(results_l)
        hands_cam_r = self._extract_hands_with_labels(results_r)
        
        # Match hands across stereo pair
        left_hand_l, left_hand_r = self._match_hand_across_cameras(hands_cam_l, hands_cam_r, "left")
        right_hand_l, right_hand_r = self._match_hand_across_cameras(hands_cam_l, hands_cam_r, "right")
        
        # Compute 3D for left hand
        left_hand_values = None
        if left_hand_l is not None and left_hand_r is not None:
            points_3d_left = self._triangulate_hand(left_hand_l, left_hand_r, frame_l.shape)
            left_hand_values = self._hand_3d_to_physical(points_3d_left)
            self._last_left_hand_3d = points_3d_left
        
        # Compute 3D for right hand
        right_hand_values = None
        if right_hand_l is not None and right_hand_r is not None:
            points_3d_right = self._triangulate_hand(right_hand_l, right_hand_r, frame_l.shape)
            right_hand_values = self._hand_3d_to_physical(points_3d_right)
            self._last_right_hand_3d = points_3d_right
        
        # Annotate frames
        annotated_l = frame_l.copy()
        annotated_r = frame_r.copy()
        
        if results_l.multi_hand_landmarks:
            for hand_landmarks in results_l.multi_hand_landmarks:
                MP_DRAW.draw_landmarks(annotated_l, hand_landmarks, MP_CONNECTIONS)
        
        if results_r.multi_hand_landmarks:
            for hand_landmarks in results_r.multi_hand_landmarks:
                MP_DRAW.draw_landmarks(annotated_r, hand_landmarks, MP_CONNECTIONS)
        
        return left_hand_values, right_hand_values, annotated_l, annotated_r
    
    def _extract_hands_with_labels(self, results):
        """Extract hands with their labels and positions."""
        hands = []
        if not results.multi_hand_landmarks:
            return hands
        
        handedness_list = getattr(results, "multi_handedness", None)
        for i, hand_lm in enumerate(results.multi_hand_landmarks):
            mp_label = handedness_list[i].classification[0].label.lower() if handedness_list else "unknown"
            
            # SWAP labels: MediaPipe's "Left" is actually user's anatomical left hand,
            # but from camera perspective (mirrored), we need to swap
            # MediaPipe "Left" -> User's RIGHT hand (camera sees it on left side)
            # MediaPipe "Right" -> User's LEFT hand (camera sees it on right side)
            if mp_label == "left":
                label = "right"  # Swap
            elif mp_label == "right":
                label = "left"   # Swap
            else:
                label = "unknown"
            
            score = handedness_list[i].classification[0].score if handedness_list else 0.0
            wrist_x = hand_lm.landmark[0].x
            hands.append({
                'landmarks': hand_lm,
                'label': label,  # Now correctly swapped
                'score': score,
                'wrist_x': wrist_x
            })
        return hands
    
    def _match_hand_across_cameras(self, hands_cam_l, hands_cam_r, target_label):
        """FAST hand matching - just take first match."""
        # Find first hand with target label in each camera (fast!)
        match_l = next((h['landmarks'] for h in hands_cam_l if h['label'] == target_label), None)
        match_r = next((h['landmarks'] for h in hands_cam_r if h['label'] == target_label), None)
        return match_l, match_r
    
    def _triangulate_hand(self, hand_l, hand_r, frame_shape):
        """FAST triangulation."""
        h, w = frame_shape[:2]
        
        # Vectorized extraction (faster!)
        pts_l = np.array([(lm.x * w, lm.y * h) for lm in hand_l.landmark], dtype=np.float32)
        pts_r = np.array([(lm.x * w, lm.y * h) for lm in hand_r.landmark], dtype=np.float32)
        
        # Vectorized triangulation
        points_3d = np.zeros((21, 3), dtype=np.float32)
        P1, P2 = self.calibration["P1"], self.calibration["P2"]
        
        for i in range(21):
            pt_4d = cv2.triangulatePoints(P1, P2, pts_l[i].reshape(2,1), pts_r[i].reshape(2,1))
            points_3d[i] = (pt_4d[:3] / pt_4d[3]).flatten()
        
        return points_3d
    
    def _hand_3d_to_physical(self, points_3d: np.ndarray) -> np.ndarray:
        """Convert 3D hand to physical values [0-100]."""
        hl = MP_HANDS.HandLandmark
        
        values = np.zeros(7, dtype=np.float32)
        values[0] = self._thumb_abduction_3d(points_3d, hl)
        values[1] = self._joint_flexion_3d(points_3d, hl.THUMB_CMC, hl.THUMB_MCP, hl.THUMB_IP, 5.0, 65.0)
        values[2] = self._joint_flexion_3d(points_3d, hl.THUMB_MCP, hl.THUMB_IP, hl.THUMB_TIP, 5.0, 115.0)
        values[3] = self._finger_curl_3d(
            points_3d, _landmark(hl, "INDEX_MCP", "INDEX_FINGER_MCP"),
            _landmark(hl, "INDEX_PIP", "INDEX_FINGER_PIP"),
            _landmark(hl, "INDEX_TIP", "INDEX_FINGER_TIP"), 5.0, 110.0
        )
        values[4] = self._finger_curl_3d(
            points_3d, _landmark(hl, "MIDDLE_MCP", "MIDDLE_FINGER_MCP"),
            _landmark(hl, "MIDDLE_PIP", "MIDDLE_FINGER_PIP"),
            _landmark(hl, "MIDDLE_TIP", "MIDDLE_FINGER_TIP"), 5.0, 110.0
        )
        values[5] = self._finger_curl_3d(
            points_3d, _landmark(hl, "RING_MCP", "RING_FINGER_MCP"),
            _landmark(hl, "RING_PIP", "RING_FINGER_PIP"),
            _landmark(hl, "RING_TIP", "RING_FINGER_TIP"), 5.0, 110.0
        )
        values[6] = self._finger_curl_3d(
            points_3d, _landmark(hl, "PINKY_MCP", "PINKY_FINGER_MCP"),
            _landmark(hl, "PINKY_PIP", "PINKY_FINGER_PIP"),
            _landmark(hl, "PINKY_TIP", "PINKY_FINGER_TIP"), 5.0, 110.0
        )
        
        return np.clip(values * 100.0, 0.0, 100.0)
    
    def _finger_curl_3d(self, points_3d, mcp, pip, tip, open_deg, closed_deg):
        """Compute finger curl from 3D positions."""
        base = points_3d[mcp.value]
        mid = points_3d[pip.value]
        fingertip = points_3d[tip.value]
        v1 = mid - base
        v2 = fingertip - mid
        return self._angle_to_ratio(v1, v2, open_deg, closed_deg)
    
    def _joint_flexion_3d(self, points_3d, root_idx, mid_idx, tip_idx, open_deg, closed_deg):
        """Compute joint flexion from 3D positions."""
        root = points_3d[root_idx.value]
        mid = points_3d[mid_idx.value]
        tip = points_3d[tip_idx.value]
        v1 = mid - root
        v2 = tip - mid
        return self._angle_to_ratio(v1, v2, open_deg, closed_deg)
    
    def _thumb_abduction_3d(self, points_3d, hl):
        """Compute thumb abduction from 3D positions."""
        wrist = points_3d[hl.WRIST.value]
        thumb_mcp = points_3d[hl.THUMB_MCP.value]
        index_landmark = _landmark(hl, "INDEX_MCP", "INDEX_FINGER_MCP")
        index_mcp = points_3d[index_landmark.value]
        
        thumb_vec = thumb_mcp - wrist
        index_vec = index_mcp - wrist
        
        angle_ratio = self._angle_to_ratio(thumb_vec, index_vec, 0.0, 90.0)
        return 1.0 - angle_ratio
    
    @staticmethod
    def _angle_to_ratio(v1, v2, open_deg, closed_deg):
        """Convert angle between 3D vectors to [0, 1] ratio."""
        angle = StereoDualHandTracker._vector_angle_3d(v1, v2)
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
    def _vector_angle_3d(v1, v2):
        """Compute angle between two 3D vectors."""
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        if v1_norm < 1e-5 or v2_norm < 1e-5:
            return 0.0
        
        cos_angle = np.clip(np.dot(v1, v2) / (v1_norm * v2_norm), -1.0, 1.0)
        return float(np.arccos(cos_angle))
    
    def _render_preview(self, frame_l: np.ndarray, frame_r: np.ndarray):
        """Display BOTH camera feeds so you can see if visible in both."""
        # Resize to 640px per camera (total 1280px width)
        h, w = frame_l.shape[:2]
        scale = 640 / w
        new_w, new_h = 640, int(h * scale)
        view_l = cv2.resize(frame_l, (new_w, new_h))
        view_r = cv2.resize(frame_r, (new_w, new_h))
        
        # Side by side
        combined = np.hstack((view_l, view_r))
        
        if self._mirror_preview:
            combined = cv2.flip(combined, 1)
        
        cv2.putText(combined, "Stereo 3D: LEFT camera | RIGHT camera (Q to exit)",
                   (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show values for BOTH hands
        labels = ["TAb", "Th1", "Th2", "Idx", "Mid", "Rng", "Pny"]
        
        if self._left_initialized:
            left_stats = " ".join(f"{name}:{val:4.1f}" for name, val in zip(labels, self._left_hand_physical))
            cv2.putText(combined, f"LEFT:  {left_stats}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(combined, "LEFT:  Not detected", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if self._right_initialized:
            right_stats = " ".join(f"{name}:{val:4.1f}" for name, val in zip(labels, self._right_hand_physical))
            cv2.putText(combined, f"RIGHT: {right_stats}", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)
        else:
            cv2.putText(combined, "RIGHT: Not detected", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        try:
            cv2.imshow(self._preview_window, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._stop_requested = True
        except cv2.error:
            self._show_preview = False
    
    def should_stop(self) -> bool:
        return self._stop_requested
    
    def close(self):
        """Release resources."""
        if self._cap_left is not None:
            self._cap_left.release()
        if self._cap_right is not None:
            self._cap_right.release()
        
        if hasattr(self._hands_left, "close"):
            self._hands_left.close()
        if hasattr(self._hands_right, "close"):
            self._hands_right.close()
        
        if self._show_preview:
            try:
                cv2.destroyWindow(self._preview_window)
            except cv2.error:
                pass


