#!/usr/bin/env python3
"""
VR Joint Calibration Wizard (Robosuite AeroPiper / teleop/gui.py compatible).

Goal:
  Record stable VR controller values for a small set of named "anchor" poses and
  bind them to your known-good robot joint targets (normalized [-1..+1]).

This script writes (into `teleop/vr_module/`):
  - vr_joint_calibration.json               (machine-readable calibration)
  - vr_joint_calibration_summary.txt        (human-readable record)
  - vr_joint_calibration_keyposes.txt       (concise key-pose record)

Important (this adaptation):
  - The optional pose preview window uses the SAME Robosuite environment as `teleop/gui.py`
    and we apply targets by setting arm joint qpos values (not AeroPiperBase actuator ctrl).

Usage:
    python teleop/vr_module/vr_joint_calibration.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: numpy.\n\n"
        "This script is intended to be run inside your AeroPiper environment.\n"
        "Try:\n"
        "  conda activate aeropiper\n"
        "  python teleop/vr_module/vr_joint_calibration.py\n"
    ) from exc

# Force unbuffered console output
sys.stdout.reconfigure(line_buffering=True)

# -----------------------------------------------------------------------------
# Path + environment setup
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
TELEOP_DIR = SCRIPT_DIR.parent
REPO_ROOT = TELEOP_DIR.parent
MODULE_DIR = SCRIPT_DIR
for path in (REPO_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Optional noise suppression: avoid warp/menagerie downloads by default.
_mj_empty = Path(tempfile.gettempdir()) / "mj_empty"
os.environ.setdefault("MUJOCO_MENAGERIE_PATH", str(_mj_empty))
_mj_empty.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MUJOCO_WARP_DISABLE", "1")

try:
    import openvr
except ImportError:  # pragma: no cover - runtime convenience
    print("=" * 60)
    print("Installing openvr (required for SteamVR access)...")
    print("=" * 60)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openvr"])
    import openvr  # type: ignore

try:
    import aeropiper as suite

    _HAS_PREVIEW = True
except Exception:  # pragma: no cover - optional preview
    suite = None  # type: ignore
    _HAS_PREVIEW = False

from vr_joint_model import (  # noqa: E402
    feature_from_pose,
    quat_average,
    quat_normalize,
    rotmat_to_quat_wxyz,
)

DEFAULT_OUT = SCRIPT_DIR / "vr_joint_calibration.json"
DEFAULT_SUMMARY = SCRIPT_DIR / "vr_joint_calibration_summary.txt"
DEFAULT_KEYPOSES = SCRIPT_DIR / "vr_joint_calibration_keyposes.txt"

POS_RANGE = 1.0
ROT_RANGE = float(np.pi)

# -----------------------------------------------------------------------------
# Timed calibration defaults (edit constants here; no CLI flags)
# -----------------------------------------------------------------------------
START_DELAY_SECS = 10.0
BETWEEN_DELAY_SECS = 10.0
REST_SECS = 3.0
POSE_SECS = 3.0
POLL_HZ = 100.0

REQUIRE_BOTH_CONTROLLERS = True
PAUSE_ON_TRACKING_LOSS = True
TRACKING_POLL_DT = 0.02
RECONNECT_SETTLE_SECS = 3.0

MAX_POSES = 14
INBETWEEN_POSES = 2
COVERAGE_POSES = 0
COVERAGE_RANGE = 0.8
COVERAGE_SEED = 0

BEEP_ENABLED = True


def _beep(freq_hz: int = 880, duration_ms: int = 120) -> None:
    """Best-effort audible cue (useful in headset)."""
    try:  # Windows
        import winsound  # type: ignore

        winsound.Beep(int(freq_hz), int(duration_ms))
    except Exception:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# OpenVR helpers
# -----------------------------------------------------------------------------
def find_controllers(vr_system: "openvr.IVRSystem") -> Tuple[Optional[int], Optional[int]]:
    controllers: List[int] = []
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        try:
            device_class = vr_system.getTrackedDeviceClass(idx)
        except Exception:
            continue
        if device_class == openvr.TrackedDeviceClass_Controller:
            controllers.append(idx)

    left_id: Optional[int] = None
    right_id: Optional[int] = None
    for idx in controllers:
        try:
            role = vr_system.getControllerRoleForTrackedDeviceIndex(idx)
        except Exception:
            role = 0
        if role == openvr.TrackedControllerRole_LeftHand:
            left_id = idx
        elif role == openvr.TrackedControllerRole_RightHand:
            right_id = idx

    remaining = [i for i in controllers if i not in {left_id, right_id}]
    if left_id is None and remaining:
        left_id = remaining.pop(0)
    if right_id is None and remaining:
        right_id = remaining.pop(0)
    return left_id, right_id


def _format_controller_role(vr_system: "openvr.IVRSystem", idx: int) -> str:
    try:
        role = vr_system.getControllerRoleForTrackedDeviceIndex(idx)
    except Exception:
        role = 0
    if role == openvr.TrackedControllerRole_LeftHand:
        return "LEFT"
    if role == openvr.TrackedControllerRole_RightHand:
        return "RIGHT"
    return "UNKNOWN"


def _print_controller_debug(vr_system: "openvr.IVRSystem") -> None:
    controllers: List[int] = []
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        try:
            if vr_system.getTrackedDeviceClass(idx) == openvr.TrackedDeviceClass_Controller:
                controllers.append(idx)
        except Exception:
            continue
    if not controllers:
        print("[DEBUG] OpenVR sees no controller-class devices.")
        return
    print("[DEBUG] OpenVR controller devices:")
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0.0, openvr.k_unMaxTrackedDeviceCount
    )
    for idx in controllers:
        p = poses[idx]
        role = _format_controller_role(vr_system, idx)
        conn = bool(getattr(p, "bDeviceIsConnected", False))
        valid = bool(getattr(p, "bPoseIsValid", False))
        print(f"  - idx={idx:2d} role={role:7s} connected={conn} pose_valid={valid}")


def openvr_pose_to_pos_quat(pose_mat) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.array([pose_mat[0][3], pose_mat[1][3], pose_mat[2][3]], dtype=np.float64)
    R = np.array(
        [
            [pose_mat[0][0], pose_mat[0][1], pose_mat[0][2]],
            [pose_mat[1][0], pose_mat[1][1], pose_mat[1][2]],
            [pose_mat[2][0], pose_mat[2][1], pose_mat[2][2]],
        ],
        dtype=np.float64,
    )
    quat = rotmat_to_quat_wxyz(R)
    return pos, quat


def compute_mean_pose(samples_pos: List[np.ndarray], samples_quat: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_arr = np.asarray(samples_pos, dtype=np.float64)
    quat_arr = np.asarray(samples_quat, dtype=np.float64)
    pos_mean = np.mean(pos_arr, axis=0)
    pos_std = np.std(pos_arr, axis=0)
    quat_mean = quat_average(quat_arr)
    return pos_mean, pos_std, quat_mean


def record_pose_samples(
    vr_system: "openvr.IVRSystem",
    left_id: Optional[int],
    right_id: Optional[int],
    duration_s: float,
    poll_hz: float,
    on_tick: Optional[Callable[[], None]] = None,
) -> Dict[str, Dict[str, object]]:
    left_id, right_id = find_controllers(vr_system)
    poll_dt = 1.0 / float(max(1e-3, poll_hz))
    active_t = 0.0
    samples_pos: Dict[str, List[np.ndarray]] = {"left": [], "right": []}
    samples_quat: Dict[str, List[np.ndarray]] = {"left": [], "right": []}

    last_len = 0
    last_status = ""
    last_status_t = 0.0
    last_rescan_t = 0.0
    needs_settle = False
    settle_until = 0.0

    while active_t < duration_s:
        poses = vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            0.0,
            openvr.k_unMaxTrackedDeviceCount,
        )
        ok_left = True
        ok_right = True
        if left_id is not None:
            p = poses[left_id]
            ok_left = bool(p.bDeviceIsConnected and p.bPoseIsValid)
        if right_id is not None:
            p = poses[right_id]
            ok_right = bool(p.bDeviceIsConnected and p.bPoseIsValid)

        if (left_id is not None and not ok_left) or (right_id is not None and not ok_right):
            if PAUSE_ON_TRACKING_LOSS:
                # Mark that when tracking returns we should wait a bit before resuming sampling.
                needs_settle = True
                now = time.time()
                if now - last_rescan_t > 1.0:
                    new_left, new_right = find_controllers(vr_system)
                    if (new_left, new_right) != (left_id, right_id):
                        left_id, right_id = new_left, new_right
                    last_rescan_t = now

                missing = []
                if left_id is not None and not ok_left:
                    missing.append("LEFT")
                if right_id is not None and not ok_right:
                    missing.append("RIGHT")
                msg = f"[PAUSED] Waiting for controller tracking: {', '.join(missing)} (timer paused)"
                if msg != last_status or (now - last_status_t) > 1.0:
                    sys.stdout.write("\r" + msg + " " * 20)
                    sys.stdout.flush()
                    last_status = msg
                    last_status_t = now
                if on_tick is not None:
                    on_tick()
                time.sleep(TRACKING_POLL_DT)
                continue

            if on_tick is not None:
                on_tick()
            time.sleep(poll_dt)
            continue

        # Tracking is OK: if we just recovered from a pause, wait a few seconds to let the user re-stabilize.
        now = time.time()
        if needs_settle:
            settle_until = now + float(max(0.0, RECONNECT_SETTLE_SECS))
            needs_settle = False
        if settle_until > now:
            rem = settle_until - now
            msg = f"[RECONNECTED] Stabilizing... {rem:4.1f}s"
            pad = max(0, last_len - len(msg))
            sys.stdout.write("\r" + msg + " " * pad)
            sys.stdout.flush()
            last_len = len(msg)
            if on_tick is not None:
                on_tick()
            time.sleep(min(TRACKING_POLL_DT, max(0.0, rem)))
            continue

        if left_id is not None:
            p = poses[left_id]
            pos, quat = openvr_pose_to_pos_quat(p.mDeviceToAbsoluteTracking)
            samples_pos["left"].append(pos)
            samples_quat["left"].append(quat_normalize(quat))
        if right_id is not None:
            p = poses[right_id]
            pos, quat = openvr_pose_to_pos_quat(p.mDeviceToAbsoluteTracking)
            samples_pos["right"].append(pos)
            samples_quat["right"].append(quat_normalize(quat))

        active_t += poll_dt

        remaining = max(0.0, duration_s - active_t)
        bar_len = 30
        filled = int(bar_len * (active_t / max(1e-9, duration_s)))
        bar = "#" * filled + "-" * (bar_len - filled)
        line = f"  Recording: [{bar}] {remaining:4.1f}s remaining"
        pad = max(0, last_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        last_len = len(line)

        if on_tick is not None:
            on_tick()
        time.sleep(poll_dt)

    sys.stdout.write("\n")

    out: Dict[str, Dict[str, object]] = {}
    for hand in ("left", "right"):
        if len(samples_pos[hand]) < 10:
            out[hand] = {"n": len(samples_pos[hand]), "ok": False}
            continue
        pos_mean, pos_std, quat_mean = compute_mean_pose(samples_pos[hand], samples_quat[hand])
        out[hand] = {
            "ok": True,
            "n": len(samples_pos[hand]),
            "pos_mean": pos_mean.tolist(),
            "pos_std": pos_std.tolist(),
            "quat_mean": quat_mean.tolist(),
        }
    return out


# -----------------------------------------------------------------------------
# Pose specs (same intent as the original calibration script)
# -----------------------------------------------------------------------------
@dataclass
class PoseSpec:
    name: str
    description: str
    duration_s: float
    robot_targets_norm: Dict[str, List[float]]  # {left:[6], right:[6]}


def default_poses(rest_s: float, other_s: float) -> List[PoseSpec]:
    return [
        PoseSpec(
            name="resting",
            description=(
                "NORMAL / RESTING reference pose.\n"
                "This is the first pose recorded and is used as the reference for features.\n"
                "You should mimic the robot pose shown in the preview window for this pose.\n"
                "Try to stay very still during the recording window."
            ),
            duration_s=rest_s,
            robot_targets_norm={
                "left": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "right": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="free_hands",
            description="FREE HANDS / RELAXED pose.\nLet your arms relax like a human when doing nothing. Try to stay steady.",
            duration_s=other_s,
            robot_targets_norm={
                "left": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                "right": [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="titanic_t_pose",
            description=(
                "TITANIC / T-POSE.\n"
                "Spread your arms out to the sides (like Titanic), palms down.\n"
                "Keep your arms horizontal and as straight as you can."
            ),
            duration_s=other_s,
            robot_targets_norm={
                "left": [-0.6, 1.0, -1.0, 0.0, 0.0, 0.0],
                "right": [0.6, 1.0, -1.0, 0.0, 0.0, 0.0],
            },
        ),
    ]


def recommended_pose_templates(other_s: float) -> List[PoseSpec]:
    return [
        PoseSpec(
            name="arms_forward_parallel",
            description=(
                "ARMS FORWARD.\n"
                "Extend both hands forward, parallel to each other.\n"
                "Try to keep your palms facing each other.\n"
                "\n"
                "Important (to learn wrist joints):\n"
                "  - Rotate wrist1 (j4) POSITIVE while holding this pose.\n"
                "  - Keep wrist2 (j5) at 0 (we are skipping j5).\n"
            ),
            duration_s=other_s,
            robot_targets_norm={
                "left": [0.0, 1.0, -1.0, 0.6, 0.0, 0.0],
                "right": [0.0, 1.0, -1.0, 0.6, 0.0, 0.0],
            },
        ),
        PoseSpec(
            name="prayer_pose",
            description=(
                "PRAYER POSE.\n"
                "Bring both hands together in front of your chest like praying.\n"
                "This is great for learning elbow/wrist changes with similar positions.\n"
                "\n"
                "Important (to learn wrist joints):\n"
                "  - Rotate wrist3 (j6) POSITIVE while holding this pose.\n"
                "  - Keep wrist2 (j5) at 0 (we are skipping j5).\n"
            ),
            duration_s=other_s,
            robot_targets_norm={
                "left": [-0.6, 1.0, 0.0, 0.0, 0.0, 0.6],
                "right": [0.6, 1.0, 0.0, 0.0, 0.0, 0.6],
            },
        ),
        PoseSpec(
            name="mid_between_rest_and_tpose",
            description=(
                "MID POSE.\n"
                "Half-way between RESTING and T-POSE (arms ~45° out).\n"
                "This improves interpolation in the middle of your workspace.\n"
                "\n"
                "Important (to learn wrist joints):\n"
                "  - Rotate wrist1 (j4) NEGATIVE and wrist3 (j6) NEGATIVE.\n"
                "  - Keep wrist2 (j5) at 0 (we are skipping j5).\n"
            ),
            duration_s=other_s,
            robot_targets_norm={
                "left": [-0.3, 0.5, -0.5, -0.6, 0.0, -0.6],
                "right": [0.3, 0.5, -0.5, -0.6, 0.0, -0.6],
            },
        ),
    ]


def extra_user_pose_templates(other_s: float) -> List[PoseSpec]:
    poses: List[PoseSpec] = []
    poses.append(
        PoseSpec(
            name="hands_inside_from_elbow",
            description=(
                "HANDS INSIDE (from elbow).\n"
                "Bring forearms inward (as if crossing in front, but keep comfortable).\n"
                "Try to keep wrists neutral (we're focusing on base rotation here)."
            ),
            duration_s=other_s,
            robot_targets_norm={"right": [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0], "left": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]},
        )
    )
    poses.append(
        PoseSpec(
            name="max_elbow_bend",
            description=(
                "MAX ELBOW BEND.\n"
                "Bend elbows as much as comfortable while keeping shoulders/base near neutral.\n"
                "Try to keep wrists neutral."
            ),
            duration_s=other_s,
            robot_targets_norm={"right": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "left": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]},
        )
    )
    poses.append(
        PoseSpec(
            name="max_elbow_and_shoulder",
            description=(
                "MAX ELBOW + SHOULDER.\n"
                "Raise/engage shoulders and bend elbows strongly (as comfortable).\n"
                "Try to keep wrists neutral."
            ),
            duration_s=other_s,
            robot_targets_norm={"right": [-0.6, -1.0, 1.0, 0.0, 0.0, 0.0], "left": [0.6, -1.0, 1.0, 0.0, 0.0, 0.0]},
        )
    )
    poses.append(
        PoseSpec(
            name="elbow_cross",
            description=(
                "ELBOW CROSS.\n"
                "Bring elbows/forearms inward across your body slightly.\n"
                "Keep motion smooth and steady; wrists neutral."
            ),
            duration_s=other_s,
            robot_targets_norm={"right": [-0.30, 0.0, 0.0, 0.0, 0.0, 0.0], "left": [0.30, 0.0, 0.0, 0.0, 0.0, 0.0]},
        )
    )
    return poses


def generate_inbetween_between_core(core: List[PoseSpec], n_total: int) -> List[PoseSpec]:
    n_total = int(max(0, n_total))
    if n_total == 0:
        return []

    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for p in core:
        l = np.asarray(p.robot_targets_norm.get("left", []), dtype=np.float64)
        r = np.asarray(p.robot_targets_norm.get("right", []), dtype=np.float64)
        if l.shape == (6,) and r.shape == (6,):
            pairs.append((l, r))
    if len(pairs) < 2:
        return []

    segs = len(pairs) - 1
    base_k = n_total // segs
    rem = n_total % segs
    out: List[PoseSpec] = []
    idx = 1
    for s in range(segs):
        k = base_k + (1 if s < rem else 0)
        if k <= 0:
            continue
        l0, r0 = pairs[s]
        l1, r1 = pairs[s + 1]
        for j in range(1, k + 1):
            a = float(j) / float(k + 1)
            ll = (1.0 - a) * l0 + a * l1
            rr = (1.0 - a) * r0 + a * r1
            out.append(
                PoseSpec(
                    name=f"inbetween_{idx:02d}",
                    description="IN-BETWEEN pose (viewer-guided).\nMimic the robot pose shown in the preview window.",
                    duration_s=float(POSE_SECS),
                    robot_targets_norm={"left": ll.tolist(), "right": rr.tolist()},
                )
            )
            idx += 1
    return out


def _lhs_unit(n: int, d: int, rng: "np.random.Generator") -> np.ndarray:
    n = int(max(1, n))
    d = int(max(1, d))
    out = np.zeros((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / float(n)
    return out


def generate_coverage_targets(n: int, r: float, rng: "np.random.Generator") -> List[Dict[str, List[float]]]:
    r = float(np.clip(r, 0.1, 1.0))
    n = int(max(0, n))
    if n == 0:
        return []

    base = np.zeros(6, dtype=np.float64)
    targets: List[Dict[str, List[float]]] = []

    amp = 0.75 * r
    for j in range(6):
        for sign in (+1.0, -1.0):
            rr = base.copy()
            ll = base.copy()
            rr[j] = float(np.clip(sign * amp, -1.0, 1.0))
            ll[j] = float(np.clip(sign * amp, -1.0, 1.0))
            ll[0] = -rr[0]
            targets.append({"left": ll.tolist(), "right": rr.tolist()})

    mixed_n = max(0, n - min(n, len(targets)))
    if mixed_n > 0:
        U = _lhs_unit(mixed_n, 6, rng)
        V = (2.0 * U - 1.0) * r
        for i in range(mixed_n):
            rr = V[i].copy()
            ll = rr.copy()
            ll[0] = -rr[0]
            targets.append({"left": ll.tolist(), "right": rr.tolist()})

    return targets[:n]


def _build_pose_list() -> List[PoseSpec]:
    core: List[PoseSpec] = default_poses(REST_SECS, POSE_SECS)
    core.extend(recommended_pose_templates(POSE_SECS))
    core.extend(extra_user_pose_templates(POSE_SECS))
    inb = generate_inbetween_between_core(core, INBETWEEN_POSES)

    poses: List[PoseSpec] = []
    poses.extend(core)
    poses.extend(inb)

    if COVERAGE_POSES > 0:
        rng = np.random.default_rng(None if COVERAGE_SEED == 0 else COVERAGE_SEED)
        candidates = generate_coverage_targets(n=COVERAGE_POSES, r=COVERAGE_RANGE, rng=rng)
        start_idx = 1
        for k, t in enumerate(candidates):
            poses.append(
                PoseSpec(
                    name=f"coverage_{start_idx + k:02d}",
                    description="COVERAGE pose (viewer-guided).\nMimic the robot pose shown in the preview window.",
                    duration_s=float(POSE_SECS),
                    robot_targets_norm={"left": list(t["left"]), "right": list(t["right"])},
                )
            )

    return poses[: int(MAX_POSES)]


# -----------------------------------------------------------------------------
# Robosuite preview helpers (optional)
# -----------------------------------------------------------------------------
def _get_joint_limit(sim, joint_name: str) -> Tuple[float, float, bool]:
    jid = sim.model.joint_name2id(joint_name)
    limited = bool(sim.model.jnt_limited[jid])
    lo, hi = sim.model.jnt_range[jid]
    if not limited:
        lo, hi = -np.pi, np.pi
    return float(lo), float(hi), limited


def _normalized_to_joint_range(norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    ratio = 0.5 * (np.clip(norm, -1.0, 1.0) + 1.0)
    return lo + ratio * (hi - lo)


def _set_joint_qvel_zero_by_addr(sim, addr) -> None:
    if isinstance(addr, (int, np.integer)):
        sim.data.qvel[int(addr)] = 0.0
    else:
        a0, a1 = addr
        sim.data.qvel[a0:a1] = 0.0


@dataclass
class PreviewRobot:
    env: object
    sim: object
    left_arm_joints: List[str]
    right_arm_joints: List[str]
    left_lo: np.ndarray
    left_hi: np.ndarray
    right_lo: np.ndarray
    right_hi: np.ndarray
    left_qvel_addrs: List[object]
    right_qvel_addrs: List[object]

    def apply_arm_targets(self, pose_targets_norm: Dict[str, List[float]]) -> None:
        l = np.asarray(pose_targets_norm.get("left", []), dtype=np.float64).reshape(-1)
        r = np.asarray(pose_targets_norm.get("right", []), dtype=np.float64).reshape(-1)
        if l.shape[0] == 6:
            lt = _normalized_to_joint_range(l, self.left_lo, self.left_hi)
            for j, q, addr in zip(self.left_arm_joints, lt, self.left_qvel_addrs):
                self.sim.data.set_joint_qpos(j, float(q))
                _set_joint_qvel_zero_by_addr(self.sim, addr)
        if r.shape[0] == 6:
            rt = _normalized_to_joint_range(r, self.right_lo, self.right_hi)
            for j, q, addr in zip(self.right_arm_joints, rt, self.right_qvel_addrs):
                self.sim.data.set_joint_qpos(j, float(q))
                _set_joint_qvel_zero_by_addr(self.sim, addr)
        self.sim.forward()

    def tick_view(self) -> None:
        viewer = getattr(self.env, "viewer", None)
        if viewer is not None and hasattr(viewer, "update"):
            try:
                viewer.update()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def create_preview_robot(env_name: str = "PickPlace") -> Optional[PreviewRobot]:
    if not _HAS_PREVIEW or suite is None:
        return None
    try:
        env = suite.make(
            env_name=str(env_name),
            robots="AeroPiper",
            env_configuration="single-robot",
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            render_camera=None,
            renderer="mjviewer",
            ignore_done=True,
        )
        env.reset()
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "set_camera"):
            env.viewer.set_camera(camera_id=-1)
        if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
            env.viewer.update()

        sim = env.sim
        robot = env.robots[0]
        arm_joints = list(robot.robot_model.arm_joints)
        arms = getattr(robot, "arms", ["right", "left"])
        split = int(len(arm_joints) / max(1, len(arms)))
        right_arm_joints = arm_joints[:split]
        left_arm_joints = arm_joints[split:]

        left_lo = np.array([_get_joint_limit(sim, j)[0] for j in left_arm_joints], dtype=np.float64)
        left_hi = np.array([_get_joint_limit(sim, j)[1] for j in left_arm_joints], dtype=np.float64)
        right_lo = np.array([_get_joint_limit(sim, j)[0] for j in right_arm_joints], dtype=np.float64)
        right_hi = np.array([_get_joint_limit(sim, j)[1] for j in right_arm_joints], dtype=np.float64)
        left_qvel_addrs = [sim.model.get_joint_qvel_addr(j) for j in left_arm_joints]
        right_qvel_addrs = [sim.model.get_joint_qvel_addr(j) for j in right_arm_joints]

        return PreviewRobot(
            env=env,
            sim=sim,
            left_arm_joints=left_arm_joints,
            right_arm_joints=right_arm_joints,
            left_lo=left_lo,
            left_hi=left_hi,
            right_lo=right_lo,
            right_hi=right_hi,
            left_qvel_addrs=left_qvel_addrs,
            right_qvel_addrs=right_qvel_addrs,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[WARN] Failed to start Robosuite preview viewer: {exc}")
        return None


def countdown_wait(secs: float, label: str, *, tick: Optional[Callable[[], None]] = None, beep: bool = True) -> None:
    secs = float(max(0.0, secs))
    t_end = time.time() + secs
    last_shown = None
    while True:
        rem = t_end - time.time()
        if rem <= 0:
            break
        rem_i = int(np.ceil(rem))
        if last_shown != rem_i:
            line = f"{label} ... {rem_i:2d}s"
            sys.stdout.write("\r" + line + " " * 10)
            sys.stdout.flush()
            if beep and rem_i in (3, 2, 1):
                _beep(700 + 80 * rem_i, 80)
            last_shown = rem_i
        if tick is not None:
            tick()
        time.sleep(0.02)
    sys.stdout.write("\r" + " " * 60 + "\r")
    sys.stdout.flush()


def main() -> None:
    print("=" * 70)
    print("VR JOINT CALIBRATION WIZARD")
    print("=" * 70)
    print("Make sure SteamVR is running and both controllers are tracked.\n")

    try:
        vr_system = openvr.init(openvr.VRApplication_Other)
    except openvr.OpenVRError as exc:
        print(f"[ERROR] Could not initialize OpenVR: {exc}")
        print("Make sure SteamVR is running and the HMD is awake.")
        return

    preview: Optional[PreviewRobot] = None
    try:
        left_id, right_id = find_controllers(vr_system)
        print(f"[OK] Left controller: {'Found' if left_id is not None else 'NOT FOUND'}")
        print(f"[OK] Right controller: {'Found' if right_id is not None else 'NOT FOUND'}\n")
        _print_controller_debug(vr_system)
        if REQUIRE_BOTH_CONTROLLERS and (left_id is None or right_id is None):
            print("[ERROR] Both controllers are required for calibration. Aborting.")
            print("Fix SteamVR tracking so BOTH controllers are detected, then rerun.")
            return
        if left_id is None and right_id is None:
            print("[ERROR] No controllers found. Aborting.")
            return

        preview = create_preview_robot(env_name="PickPlace") if _HAS_PREVIEW else None
        if preview is None:
            print("[WARN] Robosuite preview unavailable; continuing without pose preview.\n")
        else:
            print("[OK] Robosuite preview viewer opened (same setup as teleop/gui.py).\n")

        poses: List[PoseSpec] = _build_pose_list()
        calibration = {
            "version": 2,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pos_range_m": POS_RANGE,
            "rot_range_rad": ROT_RANGE,
            "poses": [],
        }

        print("\nWe will record stable controller values for each pose.")
        print("Tip: keep controllers steady; higher stability = better mapping.\n")

        # Viewer tick (~60 Hz)
        view_dt = 1.0 / 60.0
        last_view = 0.0

        def tick() -> None:
            nonlocal last_view
            if preview is None:
                return
            now = time.perf_counter()
            if now - last_view >= view_dt:
                preview.tick_view()
                last_view = now

        print(f"[INFO] Total poses to record: {len(poses)}")
        print("  (core poses first, then optional extras, then in-between)\n")

        print("[READY] When you can see SteamVR tracking (and the preview window, if enabled), press ENTER to begin.")
        input()
        print(f"\n[INFO] Starting timed calibration in {int(START_DELAY_SECS)}s. Put on headset and get into POSE 1.\n")
        countdown_wait(START_DELAY_SECS, "Initial setup (POSE 1)", tick=tick, beep=BEEP_ENABLED)

        for i, pose in enumerate(poses):
            print("-" * 70)
            print(f"POSE {i + 1}/{len(poses)}: {pose.name.upper()}")
            print("-" * 70)
            print(pose.description)
            print("\nRobot targets (normalized [-1..+1])")
            print(f"  LEFT : {pose.robot_targets_norm['left']}")
            print(f"  RIGHT: {pose.robot_targets_norm['right']}\n")

            if preview is not None:
                preview.apply_arm_targets(pose.robot_targets_norm)
                tick()

            if i > 0:
                countdown_wait(BETWEEN_DELAY_SECS, "Move into NEXT pose", tick=tick, beep=BEEP_ENABLED)

            print("  RECORDING NOW (hold steady)...\n")
            if BEEP_ENABLED:
                _beep(1000, 150)

            vr_stats = record_pose_samples(
                vr_system,
                left_id,
                right_id,
                pose.duration_s,
                POLL_HZ,
                on_tick=tick,
            )
            if BEEP_ENABLED:
                _beep(600, 120)

            entry = {
                "name": pose.name,
                "description": pose.description,
                "duration_s": pose.duration_s,
                "robot_targets_norm": pose.robot_targets_norm,
                "vr": vr_stats,
            }
            calibration["poses"].append(entry)

            for hand in ("left", "right"):
                if vr_stats.get(hand, {}).get("ok", False):
                    std = np.asarray(vr_stats[hand]["pos_std"], dtype=np.float64)
                    print(f"  {hand.upper()}: n={vr_stats[hand]['n']} pos_std(max)={float(np.max(std)):.4f} m")
                else:
                    print(f"  {hand.upper()}: NO DATA (n={vr_stats.get(hand, {}).get('n', 0)})")
            print()

        # Build features relative to RESTING
        poses_out = calibration["poses"]
        ref_pose = next((p for p in poses_out if p.get("name") == "resting"), None)
        if ref_pose is None:
            print("[ERROR] No 'resting' pose recorded. Aborting save.")
            return

        for hand in ("left", "right"):
            if not ref_pose["vr"].get(hand, {}).get("ok", False):
                print(f"[ERROR] RESTING pose missing/invalid for {hand.upper()} controller. Aborting save.")
                return

        ref_pos = {h: np.asarray(ref_pose["vr"][h]["pos_mean"], dtype=np.float64) for h in ("left", "right")}
        ref_quat = {h: quat_normalize(np.asarray(ref_pose["vr"][h]["quat_mean"], dtype=np.float64)) for h in ("left", "right")}

        for p in poses_out:
            feats: Dict[str, List[float]] = {}
            for hand in ("left", "right"):
                v = p.get("vr", {}).get(hand, {})
                if not v or not v.get("ok", False):
                    continue
                pos_mean = np.asarray(v["pos_mean"], dtype=np.float64)
                quat_mean = quat_normalize(np.asarray(v["quat_mean"], dtype=np.float64))
                feats[hand] = feature_from_pose(pos_mean, quat_mean, ref_pos[hand], ref_quat[hand], POS_RANGE, ROT_RANGE).tolist()
            p["features"] = feats

        # Save JSON
        DEFAULT_OUT.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

        # Save summary TXT
        lines: List[str] = []
        lines.append("VR JOINT CALIBRATION SUMMARY")
        lines.append(f"Created: {calibration['created']}")
        lines.append(f"File: {DEFAULT_OUT}")
        lines.append("")
        lines.append("All robot targets are normalized [-1..+1] in order:")
        lines.append("  [base, shoulder, elbow, wrist1, wrist2, wrist3]")
        lines.append("")
        for p in poses_out:
            lines.append(f"POSE: {p['name']}")
            for hand in ("left", "right"):
                tgt = p["robot_targets_norm"].get(hand)
                v = p.get("vr", {}).get(hand, {})
                feat = p.get("features", {}).get(hand)
                if not v or not v.get("ok", False):
                    lines.append(f"  {hand.upper()}: NO DATA")
                    continue
                std = np.asarray(v["pos_std"], dtype=np.float64)
                lines.append(f"  {hand.upper()}: target={tgt}")
                lines.append(f"    vr_pos_mean={np.asarray(v['pos_mean'])}")
                lines.append(f"    vr_quat_mean={np.asarray(v['quat_mean'])}")
                lines.append(f"    pos_std_max={float(np.max(std)):.6f} m, n={v.get('n', 0)}")
                lines.append(f"    feature(dpos,drot)={feat}")
            lines.append("")
        DEFAULT_SUMMARY.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Save key poses record
        key_names = ("resting", "free_hands", "titanic_t_pose")
        key_lines: List[str] = []
        key_lines.append("VR JOINT CALIBRATION - KEY POSES RECORD")
        key_lines.append(f"Created: {calibration['created']}")
        key_lines.append(f"Calibration JSON: {DEFAULT_OUT}")
        key_lines.append("")
        key_lines.append("Joint order:")
        key_lines.append("  [j1(base rot), j2(shoulder), j3(elbow), j4(wrist1 rot), j5(wrist2), j6(wrist3 rot)]")
        key_lines.append("")
        for p in poses_out:
            if p.get("name") not in key_names:
                continue
            key_lines.append(f"POSE: {p['name']}")
            for hand in ("left", "right"):
                tgt = p["robot_targets_norm"].get(hand)
                v = p.get("vr", {}).get(hand, {})
                if not v or not v.get("ok", False):
                    key_lines.append(f"  {hand.upper()}: NO DATA")
                    continue
                std = np.asarray(v["pos_std"], dtype=np.float64)
                key_lines.append(f"  {hand.upper()}: target_norm={tgt}")
                key_lines.append(f"    vr_pos_mean={np.asarray(v['pos_mean'])}")
                key_lines.append(f"    vr_quat_mean(wxyz)={np.asarray(v['quat_mean'])}")
                key_lines.append(f"    pos_std_max={float(np.max(std)):.6f} m, n={v.get('n', 0)}")
            key_lines.append("")
        DEFAULT_KEYPOSES.write_text("\n".join(key_lines) + "\n", encoding="utf-8")

        print("=" * 70)
        print("CALIBRATION SAVED")
        print("=" * 70)
        print(f"[OK] JSON   : {DEFAULT_OUT}")
        print(f"[OK] Summary: {DEFAULT_SUMMARY}")
        print(f"[OK] KeyPoses: {DEFAULT_KEYPOSES}")
        print("\nNext: run the VR controller:")
        print("  python teleop/vr_control.py")

    finally:
        try:
            openvr.shutdown()
        except Exception:
            pass
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    main()


