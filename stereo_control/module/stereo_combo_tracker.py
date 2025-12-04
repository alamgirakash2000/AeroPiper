"""
Stereo 3D Arm + Hand Combo Tracker

Tracks both arm (shoulder, elbow, wrist) and hand (fingers) in true 3D
using two calibrated cameras.

This provides view-angle invariant control for all 13 DOFs:
- 6 arm joints (base yaw, shoulder pitch/roll, elbow, wrist pitch/roll)
- 7 hand DOFs (thumb abduction, thumb joints, 4 fingers)
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

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise ImportError("Ultralytics required: pip install ultralytics") from exc

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning, message="SymbolDatabase.GetPrototype")

MP_HANDS = mp.solutions.hands
MP_POSE = mp.solutions.pose
MP_DRAW = mp.solutions.drawing_utils
MP_HAND_CONNECTIONS = MP_HANDS.HAND_CONNECTIONS
MP_POSE_CONNECTIONS = MP_POSE.POSE_CONNECTIONS


def _landmark(hl, *names):
    """Return first available landmark attribute."""
    for name in names:
        if hasattr(hl, name):
            return getattr(hl, name)
    raise AttributeError(f"HandLandmark has none of: {', '.join(names)}")


class StereoComboTracker:
    """Track arm and hand in 3D using stereo vision."""
    
    def __init__(
        self,
        calibration: dict,
        left_camera_index: int = 0,
        right_camera_index: int = 1,
        show_preview: bool = True,
        mirror_preview: bool = True,
        arm_smoothing: float = 0.4,
        hand_smoothing: float = 0.35,
        max_step: float = 0.35,
        yolo_model: str = "yolov8n-pose.pt",
        yolo_frame_skip: int = 1,
    ):
        """
        Args:
            calibration: Stereo calibration from stereo_calibration.py
            left_camera_index: Left camera index
            right_camera_index: Right camera index
            show_preview: Show preview window
            mirror_preview: Mirror preview
            arm_smoothing: EMA for arm (0-1)
            hand_smoothing: EMA for hand (0-1)
            max_step: Max arm jump per frame
            yolo_model: YOLO pose model path
            yolo_frame_skip: Process every Nth frame with YOLO
        """
        self.calibration = calibration
        self._show_preview = show_preview
        self._mirror_preview = mirror_preview
        self._arm_alpha = float(np.clip(arm_smoothing, 0.0, 1.0))
        self._hand_alpha = float(np.clip(hand_smoothing, 0.0, 1.0))
        self._max_step = max_step
        self._yolo_frame_skip = max(0, int(yolo_frame_skip))
        
        # Open cameras
        self._cap_left = self._open_camera(left_camera_index)
        self._cap_right = self._open_camera(right_camera_index)
        
        # YOLO for pose (arm tracking) with GPU if available
        self._yolo = YOLO(yolo_model)
        self._yolo.to('cuda' if self._check_cuda() else 'cpu')  # Use GPU
        self._yolo_counter = 0
        self._last_yolo_keypoints_left = None
        self._last_yolo_keypoints_right = None
        
        # MediaPipe for hands - LITE for speed
        # Set max_num_hands=2 to detect both hands, then filter to left only
        self._hands_left = MP_HANDS.Hands(
            static_image_mode=False,
            model_complexity=0,  # LITE = faster
            max_num_hands=2,  # Detect both hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hands_right = MP_HANDS.Hands(
            static_image_mode=False,
            model_complexity=0,  # LITE = faster
            max_num_hands=2,  # Detect both hands
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        # State
        self._arm_physical = np.zeros(6, dtype=np.float32)
        self._hand_physical = np.zeros(7, dtype=np.float32)
        self._arm_initialized = False
        self._hand_initialized = False
        self._stop_requested = False
        self._last_detection_time = 0.0
        
        # For visualization
        self._last_arm_3d = None
        self._last_hand_3d = None
        self._preview_window = "Stereo 3D Arm + Hand Tracking"
        
        # Hand tracking persistence
        self._last_valid_hand_left = None
        self._last_valid_hand_right = None
        self._hand_frames_since_detection = 0
        self._max_hand_frames_to_persist = 10
    
    def _check_cuda(self):
        """Check if CUDA/GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
        
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
    
    def update(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update tracking.
        
        Returns:
            (arm_values [6], hand_values [7]) both in normalized/physical units
        """
        if self._stop_requested:
            return self._arm_physical.copy(), self._hand_physical.copy()
        
        # Read frames
        ret_l, frame_l = self._cap_left.read()
        ret_r, frame_r = self._cap_right.read()
        
        if not ret_l or not ret_r:
            self._stop_requested = True
            return self._arm_physical.copy(), self._hand_physical.copy()
        
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
        
        # Track arm and hand
        arm_values, annotated_l_arm, annotated_r_arm = self._track_arm_3d(
            frame_l_rect, frame_r_rect
        )
        hand_values, annotated_l_hand, annotated_r_hand = self._track_hand_3d(
            frame_l_rect, frame_r_rect
        )
        
        now = time.time()
        
        # Update arm state
        if arm_values is not None:
            if not self._arm_initialized:
                self._arm_physical = arm_values
                self._arm_initialized = True
            else:
                # Apply max_step clamping
                delta = arm_values - self._arm_physical
                delta = np.clip(delta, -self._max_step, self._max_step)
                new_values = self._arm_physical + delta
                self._arm_physical = (
                    self._arm_alpha * new_values 
                    + (1.0 - self._arm_alpha) * self._arm_physical
                )
            self._last_detection_time = now
        
        # Update hand state
        if hand_values is not None:
            if not self._hand_initialized:
                self._hand_physical = hand_values
                self._hand_initialized = True
            else:
                self._hand_physical = (
                    self._hand_alpha * hand_values
                    + (1.0 - self._hand_alpha) * self._hand_physical
                )
            self._last_detection_time = now
        
        # Show preview
        if self._show_preview:
            # Combine arm and hand annotations
            display_l = annotated_l_arm if annotated_l_arm is not None else frame_l_rect
            display_r = annotated_r_arm if annotated_r_arm is not None else frame_r_rect
            
            if annotated_l_hand is not None:
                display_l = annotated_l_hand
            if annotated_r_hand is not None:
                display_r = annotated_r_hand
            
            self._render_preview(display_l, display_r)
        
        return self._arm_physical.copy(), self._hand_physical.copy()
    
    def _track_arm_3d(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Track arm in 3D using YOLO pose."""
        # Skip frames for performance
        self._yolo_counter += 1
        if self._yolo_counter <= self._yolo_frame_skip:
            # Use cached keypoints
            if self._last_yolo_keypoints_left is None:
                return None, None, None
            kpts_l = self._last_yolo_keypoints_left
            kpts_r = self._last_yolo_keypoints_right
        else:
            self._yolo_counter = 0
            # Run YOLO on both frames
            results_l = self._yolo(frame_l, verbose=False)
            results_r = self._yolo(frame_r, verbose=False)
            
            if not results_l or not results_l[0].keypoints:
                return None, None, None
            if not results_r or not results_r[0].keypoints:
                return None, None, None
            
            kpts_l = results_l[0].keypoints.xy[0].cpu().numpy()
            kpts_r = results_r[0].keypoints.xy[0].cpu().numpy()
            
            self._last_yolo_keypoints_left = kpts_l
            self._last_yolo_keypoints_right = kpts_r
        
        # Triangulate arm joints to 3D
        # YOLO keypoints: [nose, eyes, ears, shoulders(5,6), elbows(7,8), wrists(9,10), ...]
        # We want: shoulder, elbow, wrist (left side only)
        required_indices = [5, 7, 9]  # Left shoulder, elbow, wrist
        
        points_3d = []
        for idx in required_indices:
            if idx < len(kpts_l) and idx < len(kpts_r):
                pt_2d_l = kpts_l[idx]
                pt_2d_r = kpts_r[idx]
                
                # Triangulate
                pt_4d = cv2.triangulatePoints(
                    self.calibration["P1"],
                    self.calibration["P2"],
                    pt_2d_l.reshape(2, 1),
                    pt_2d_r.reshape(2, 1),
                )
                pt_3d = pt_4d[:3] / pt_4d[3]
                points_3d.append(pt_3d.flatten())
            else:
                return None, None, None
        
        points_3d = np.array(points_3d)  # Shape: (3, 3) - shoulder, elbow, wrist
        self._last_arm_3d = points_3d
        
        # Convert 3D arm points to normalized control values
        arm_values = self._arm_3d_to_normalized(points_3d)
        
        # Annotate frames
        annotated_l = frame_l.copy()
        annotated_r = frame_r.copy()
        
        for idx in required_indices:
            pt_l = tuple(kpts_l[idx].astype(int))
            pt_r = tuple(kpts_r[idx].astype(int))
            cv2.circle(annotated_l, pt_l, 5, (0, 255, 0), -1)
            cv2.circle(annotated_r, pt_r, 5, (0, 255, 0), -1)
        
        # Draw skeleton
        cv2.line(annotated_l, tuple(kpts_l[5].astype(int)), tuple(kpts_l[7].astype(int)), (0, 255, 0), 2)
        cv2.line(annotated_l, tuple(kpts_l[7].astype(int)), tuple(kpts_l[9].astype(int)), (0, 255, 0), 2)
        cv2.line(annotated_r, tuple(kpts_r[5].astype(int)), tuple(kpts_r[7].astype(int)), (0, 255, 0), 2)
        cv2.line(annotated_r, tuple(kpts_r[7].astype(int)), tuple(kpts_r[9].astype(int)), (0, 255, 0), 2)
        
        return arm_values, annotated_l, annotated_r
    
    def _arm_3d_to_normalized(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Convert 3D arm points to normalized control values [-1, 1].
        
        Points: [shoulder, elbow, wrist] in 3D space (mm)
        Returns: [base_yaw, shoulder_pitch, shoulder_roll, elbow_flex, wrist_pitch, wrist_roll]
        """
        shoulder = points_3d[0]
        elbow = points_3d[1]
        wrist = points_3d[2]
        
        # Compute vectors
        upper_arm = elbow - shoulder
        forearm = wrist - elbow
        
        # Base yaw (rotation around vertical axis)
        base_yaw = np.arctan2(upper_arm[0], upper_arm[2])  # X/Z
        base_yaw_norm = np.clip(base_yaw / np.pi, -1.0, 1.0)
        
        # Shoulder pitch (up/down)
        shoulder_pitch = np.arctan2(-upper_arm[1], np.sqrt(upper_arm[0]**2 + upper_arm[2]**2))
        shoulder_pitch_norm = np.clip(shoulder_pitch / (np.pi/2), -1.0, 1.0)
        
        # Shoulder roll (left/right)
        shoulder_roll = np.arctan2(upper_arm[0], upper_arm[1])
        shoulder_roll_norm = np.clip(shoulder_roll / (np.pi/2), -1.0, 1.0)
        
        # Elbow flexion (angle between upper arm and forearm)
        cos_angle = np.dot(upper_arm, forearm) / (np.linalg.norm(upper_arm) * np.linalg.norm(forearm) + 1e-6)
        elbow_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        elbow_norm = np.clip((elbow_angle - np.pi/2) / (np.pi/2), -1.0, 1.0)
        
        # Wrist pitch and roll (simplified - could use hand orientation if available)
        wrist_pitch_norm = 0.0
        wrist_roll_norm = 0.0
        
        return np.array([
            base_yaw_norm,
            shoulder_pitch_norm,
            shoulder_roll_norm,
            elbow_norm,
            wrist_pitch_norm,
            wrist_roll_norm,
        ], dtype=np.float32)
    
    def _track_hand_3d(
        self, frame_l: np.ndarray, frame_r: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Track hand in 3D (same as stereo_hand_tracker)."""
        rgb_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2RGB)
        rgb_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
        
        results_l = self._hands_left.process(rgb_l)
        results_r = self._hands_right.process(rgb_r)
        
        # CRITICAL FIX: Ensure BOTH cameras track the SAME hand
        hands_l = []
        if results_l.multi_hand_landmarks:
            handedness_l = getattr(results_l, "multi_handedness", None)
            for i, hand_lm in enumerate(results_l.multi_hand_landmarks):
                label = handedness_l[i].classification[0].label.lower() if handedness_l else "unknown"
                wrist_x = hand_lm.landmark[0].x
                hands_l.append((hand_lm, label, wrist_x))
        
        hands_r = []
        if results_r.multi_hand_landmarks:
            handedness_r = getattr(results_r, "multi_handedness", None)
            for i, hand_lm in enumerate(results_r.multi_hand_landmarks):
                label = handedness_r[i].classification[0].label.lower() if handedness_r else "unknown"
                wrist_x = hand_lm.landmark[0].x
                hands_r.append((hand_lm, label, wrist_x))
        
        # Match same hand across both views
        landmarks_l = None
        landmarks_r = None
        
        if hands_l and hands_r:
            best_match_score = -1
            for hand_l, label_l, pos_l in hands_l:
                for hand_r, label_r, pos_r in hands_r:
                    score = 0
                    # CORRECTED: MediaPipe "Right" = user's LEFT hand (swapped)
                    if label_l == "right" and label_r == "right":
                        score += 10
                    pos_similarity = 1.0 - abs(pos_l - pos_r)
                    score += pos_similarity * 5
                    if score > best_match_score:
                        best_match_score = score
                        landmarks_l = hand_l
                        landmarks_r = hand_r
        
        # Apply persistence
        if landmarks_l is not None and landmarks_r is not None:
            self._last_valid_hand_left = landmarks_l
            self._last_valid_hand_right = landmarks_r
            self._hand_frames_since_detection = 0
        elif self._last_valid_hand_left is not None and self._last_valid_hand_right is not None and self._hand_frames_since_detection < self._max_hand_frames_to_persist:
            landmarks_l = self._last_valid_hand_left
            landmarks_r = self._last_valid_hand_right
            self._hand_frames_since_detection += 1
        else:
            return None, None, None
        
        if landmarks_l is None or landmarks_r is None:
            return None, None, None
        
        # Get 2D points
        points_2d_l = np.array([(lm.x, lm.y) for lm in landmarks_l.landmark], dtype=np.float32)
        points_2d_r = np.array([(lm.x, lm.y) for lm in landmarks_r.landmark], dtype=np.float32)
        
        # Convert to pixels
        h, w = frame_l.shape[:2]
        points_2d_l_px = points_2d_l * [w, h]
        points_2d_r_px = points_2d_r * [w, h]
        
        # Triangulate
        points_3d = self._triangulate_points(points_2d_l_px, points_2d_r_px)
        self._last_hand_3d = points_3d
        
        # Compute physical values
        hand_values = self._hand_3d_to_physical(points_3d)
        
        # Annotate
        annotated_l = frame_l.copy()
        annotated_r = frame_r.copy()
        MP_DRAW.draw_landmarks(annotated_l, landmarks_l, MP_HAND_CONNECTIONS)
        MP_DRAW.draw_landmarks(annotated_r, landmarks_r, MP_HAND_CONNECTIONS)
        
        return hand_values, annotated_l, annotated_r
    
    def _triangulate_points(self, points_2d_l: np.ndarray, points_2d_r: np.ndarray) -> np.ndarray:
        """Triangulate 2D points to 3D."""
        P1 = self.calibration["P1"]
        P2 = self.calibration["P2"]
        
        num_points = len(points_2d_l)
        points_3d = np.zeros((num_points, 3), dtype=np.float32)
        
        for i in range(num_points):
            pt_4d = cv2.triangulatePoints(
                P1, P2,
                points_2d_l[i].reshape(2, 1),
                points_2d_r[i].reshape(2, 1),
            )
            pt_3d = pt_4d[:3] / pt_4d[3]
            points_3d[i] = pt_3d.flatten()
        
        return points_3d
    
    def _hand_3d_to_physical(self, points_3d: np.ndarray) -> np.ndarray:
        """Convert 3D hand landmarks to physical values [0-100]."""
        hl = MP_HANDS.HandLandmark
        
        values = np.zeros(7, dtype=np.float32)
        values[0] = self._thumb_abduction_3d(points_3d, hl)
        values[1] = self._joint_flexion_3d(
            points_3d, hl.THUMB_CMC, hl.THUMB_MCP, hl.THUMB_IP, 5.0, 65.0
        )
        values[2] = self._joint_flexion_3d(
            points_3d, hl.THUMB_MCP, hl.THUMB_IP, hl.THUMB_TIP, 5.0, 115.0
        )
        values[3] = self._finger_curl_3d(
            points_3d,
            _landmark(hl, "INDEX_MCP", "INDEX_FINGER_MCP"),
            _landmark(hl, "INDEX_PIP", "INDEX_FINGER_PIP"),
            _landmark(hl, "INDEX_TIP", "INDEX_FINGER_TIP"),
            5.0, 110.0
        )
        values[4] = self._finger_curl_3d(
            points_3d,
            _landmark(hl, "MIDDLE_MCP", "MIDDLE_FINGER_MCP"),
            _landmark(hl, "MIDDLE_PIP", "MIDDLE_FINGER_PIP"),
            _landmark(hl, "MIDDLE_TIP", "MIDDLE_FINGER_TIP"),
            5.0, 110.0
        )
        values[5] = self._finger_curl_3d(
            points_3d,
            _landmark(hl, "RING_MCP", "RING_FINGER_MCP"),
            _landmark(hl, "RING_PIP", "RING_FINGER_PIP"),
            _landmark(hl, "RING_TIP", "RING_FINGER_TIP"),
            5.0, 110.0
        )
        values[6] = self._finger_curl_3d(
            points_3d,
            _landmark(hl, "PINKY_MCP", "PINKY_FINGER_MCP"),
            _landmark(hl, "PINKY_PIP", "PINKY_FINGER_PIP"),
            _landmark(hl, "PINKY_TIP", "PINKY_FINGER_TIP"),
            5.0, 110.0
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
        angle = StereoComboTracker._vector_angle_3d(v1, v2)
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
    
    @staticmethod
    def _select_left_hand(results):
        """
        Select left hand from MediaPipe results.
        
        TRUSTS MediaPipe's "Left" label - this is the user's anatomical left hand.
        """
        if not results.multi_hand_landmarks:
            return None
        
        handedness_list = getattr(results, "multi_handedness", None)
        
        if not handedness_list or len(results.multi_hand_landmarks) == 1:
            # Only one hand detected, assume it's the left
            return results.multi_hand_landmarks[0]
        
        # Multiple hands detected - TRUST MediaPipe's label
        best_left_hand = None
        best_left_score = 0.0
        
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, handedness_list
        ):
            label = handedness.classification[0].label.lower()
            score = handedness.classification[0].score
            
            # Find hand with "Left" label and highest confidence
            if label == "left" and score > best_left_score:
                best_left_score = score
                best_left_hand = hand_landmarks
        
        # Return best left hand, or first hand if no "Left" label found
        return best_left_hand if best_left_hand is not None else results.multi_hand_landmarks[0]
    
    def _render_preview(self, frame_l: np.ndarray, frame_r: np.ndarray):
        """Display stereo preview."""
        if not self._show_preview:
            return
        
        combined = np.hstack((frame_l, frame_r))
        
        if self._mirror_preview:
            combined = cv2.flip(combined, 1)
        
        cv2.putText(
            combined,
            "Stereo 3D Arm + Hand (Q/ESC to exit)",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        
        # Show values
        arm_labels = ["BY", "SP", "SR", "EF", "WP", "WR"]
        hand_labels = ["TAb", "Th1", "Th2", "Idx", "Mid", "Rng", "Pny"]
        
        arm_stats = " ".join(f"{name}:{val:+4.2f}" for name, val in zip(arm_labels, self._arm_physical))
        hand_stats = " ".join(f"{name}:{val:4.1f}" for name, val in zip(hand_labels, self._hand_physical))
        
        cv2.putText(combined, f"Arm:  {arm_stats}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(combined, f"Hand: {hand_stats}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if self._last_arm_3d is not None and self._last_hand_3d is not None:
            cv2.putText(combined, "3D TRACKING ACTIVE", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        try:
            cv2.imshow(self._preview_window, combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self._stop_requested = True
        except cv2.error:
            self._show_preview = False
    
    def should_stop(self) -> bool:
        """Check if should stop."""
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

