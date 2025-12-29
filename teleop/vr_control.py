#!/usr/bin/env python3
"""
Dual-arm AeroPiper teleoperation driven by SteamVR (OpenVR) controllers using a calibrated VR->joint model.

IMPORTANT (this adaptation):
  - This script now controls the SAME Robosuite AeroPiper setup as `teleop/gui.py`.
  - We drive the robot by writing to Robosuite's `env.sim` (joint qpos + gripper actuator ctrl),
    rather than using `envs.AeroPiperBase` / direct MuJoCo actuator targets.

How to use:
  1) Run calibration (records anchor poses to `teleop/vr_module/vr_joint_calibration.json`):
       python teleop/vr_module/vr_joint_calibration.py
  2) Run VR controller:
       python teleop/vr_control.py

Notes:
  - This script captures a *runtime* RESTING reference at startup (few seconds).
    Start it while holding your RESTING pose.
  - Trigger controls the gripper (same intent as `teleop/gui.py`: 0=open, 1=close).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: numpy.\n\n"
        "This script is intended to be run inside your AeroPiper environment.\n"
        "Try:\n"
        "  conda activate aeropiper\n"
        "  python teleop/vr_control.py\n"
    ) from exc

# Force unbuffered console output for timely status logs
sys.stdout.reconfigure(line_buffering=True)

try:
    import openvr
except ImportError:  # pragma: no cover - runtime convenience
    print("=" * 60)
    print("Installing openvr (required for SteamVR access)...")
    print("=" * 60)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openvr"])
    import openvr  # type: ignore

# -----------------------------------------------------------------------------
# Path + environment setup
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VR_MODULE_DIR = SCRIPT_DIR / "vr_module"
for path in (REPO_ROOT, VR_MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Optional noise suppression: avoid warp/menagerie downloads by default.
_mj_empty = Path(tempfile.gettempdir()) / "mj_empty"
os.environ.setdefault("MUJOCO_MENAGERIE_PATH", str(_mj_empty))
_mj_empty.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MUJOCO_WARP_DISABLE", "1")

import aeropiper as suite  # noqa: E402
from vr_joint_model import (  # noqa: E402
    HandReference,
    VRJointMapper,
    build_dataset,
    load_calibration,
    quat_average,
    quat_normalize,
    rotmat_to_quat_wxyz,
)

# -----------------------------------------------------------------------------
# Runtime tuning
# -----------------------------------------------------------------------------
CONTROLLER_RESCAN_INTERVAL_S = 1.0
OPENVR_DEBUG_PRINT_INTERVAL_S = 5.0


# -----------------------------------------------------------------------------
# OpenVR helpers (unchanged)
# -----------------------------------------------------------------------------
def find_controllers(vr_system: "openvr.IVRSystem") -> Tuple[Optional[int], Optional[int]]:
    """
    Best-effort (left_id, right_id).

    SteamVR can show controllers as "connected" while OpenVR doesn't provide valid poses yet.
    Also, left/right roles are sometimes missing/unknown in OpenVR. We therefore:
      1) Prefer role-based mapping when available
      2) Fall back to "first two controller-class devices" if roles are missing
    """
    controllers: list[int] = []
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
    controllers: list[int] = []
    for idx in range(openvr.k_unMaxTrackedDeviceCount):
        try:
            if vr_system.getTrackedDeviceClass(idx) == openvr.TrackedDeviceClass_Controller:
                controllers.append(idx)
        except Exception:
            continue
    if not controllers:
        print("[DEBUG] OpenVR sees no controller-class devices.")
        return
    poses = vr_system.getDeviceToAbsoluteTrackingPose(
        openvr.TrackingUniverseStanding, 0.0, openvr.k_unMaxTrackedDeviceCount
    )
    print("[DEBUG] OpenVR controller devices:")
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


def poll_controller(
    vr_system: "openvr.IVRSystem",
    controller_id: Optional[int],
    poses,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, str]:
    """
    Returns (pos, quat, trigger01, status)
    status ∈ {"OK","DISCONNECTED","NO TRACKING","NOT FOUND"}
    """
    if controller_id is None:
        return None, None, 0.0, "NOT FOUND"
    pose = poses[controller_id]
    if not pose.bDeviceIsConnected:
        return None, None, 0.0, "DISCONNECTED"
    if not pose.bPoseIsValid:
        return None, None, 0.0, "NO TRACKING"

    pos, quat = openvr_pose_to_pos_quat(pose.mDeviceToAbsoluteTracking)
    trigger = 0.0
    success, state = vr_system.getControllerState(controller_id)
    if success and len(state.rAxis) > 1:
        trigger = float(np.clip(state.rAxis[1].x, 0.0, 1.0))
    return pos, quat_normalize(quat), trigger, "OK"


# -----------------------------------------------------------------------------
# Robosuite (teleop/gui.py compatible) robot control helpers
# -----------------------------------------------------------------------------
def _get_joint_limit(sim, joint_name: str) -> Tuple[float, float, bool]:
    """Returns (lo, hi, limited) for a MuJoCo joint name."""
    jid = sim.model.joint_name2id(joint_name)
    limited = bool(sim.model.jnt_limited[jid])
    lo, hi = sim.model.jnt_range[jid]
    if not limited:
        lo, hi = -np.pi, np.pi
    return float(lo), float(hi), limited


def _normalized_to_joint_range(norm: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Map normalized [-1,1] -> [lo,hi] elementwise."""
    ratio = 0.5 * (np.clip(norm, -1.0, 1.0) + 1.0)
    return lo + ratio * (hi - lo)


def _get_actuator_ctrlrange(sim, actuator_name: str) -> Tuple[int, float, float]:
    aid = sim.model.actuator_name2id(actuator_name)
    lo, hi = sim.model.actuator_ctrlrange[aid]
    return int(aid), float(lo), float(hi)


def _build_gripper_actuator_map(sim, gripper_actuator_names: List[str]) -> Optional[dict]:
    """
    Builds actuator control specs for a gripper:
      - identifies thumb abduction actuator (thumb*abd*)
      - returns it separately so it can be pinned to max
      - returns remaining actuators as list[(aid, lo, hi)] that are controlled by the trigger
    """
    names = list(gripper_actuator_names)
    if not names:
        return None

    def _is_thumb_abd(n: str) -> bool:
        s = n.lower()
        return ("thumb" in s) and ("abd" in s)

    thumb_abd_name = next((n for n in names if _is_thumb_abd(n)), None)
    if thumb_abd_name is None:
        thumb_abd_name = next((n for n in names if "abd" in n.lower()), None)

    thumb_abd = None
    if thumb_abd_name is not None:
        aid, lo, hi = _get_actuator_ctrlrange(sim, thumb_abd_name)
        thumb_abd = (aid, lo, hi)
        names = [n for n in names if n != thumb_abd_name]

    controlled = []
    for n in names:
        aid, lo, hi = _get_actuator_ctrlrange(sim, n)
        controlled.append((aid, lo, hi))

    return {"thumb_abd": thumb_abd, "controlled": controlled}


def _gripper_extra_flex_joints(gripper_joint_names: List[str]) -> List[str]:
    """
    Returns gripper joints that are not actuated but should still flex for a full curl.
    Matches `teleop/gui.py` behavior.
    """
    out: List[str] = []
    for j in gripper_joint_names:
        jl = j.lower()
        if ("_pip" in jl) or ("_dip" in jl) or ("thumb_ip" in jl):
            out.append(j)
    return out


def _set_joint_qvel_zero_by_addr(sim, addr) -> None:
    if isinstance(addr, (int, np.integer)):
        sim.data.qvel[int(addr)] = 0.0
    else:
        a0, a1 = addr
        sim.data.qvel[a0:a1] = 0.0


def write_inline_status(line: str, last_len: int) -> int:
    padding = max(0, last_len - len(line))
    sys.stdout.write("\r" + line + " " * padding)
    sys.stdout.flush()
    return len(line)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Control AeroPiper (Robosuite sim) using a calibrated VR->joint model.")

    # Robosuite env (match teleop/gui.py defaults)
    p.add_argument("--env", type=str, default="PickPlace", help="Robosuite environment name (default: PickPlace).")
    p.add_argument("--no-viewer", action="store_true", help="Run without launching the Robosuite mjviewer window.")

    # VR polling
    p.add_argument("--poll-interval", type=float, default=0.005, help="Seconds between SteamVR polls (default: 5ms).")
    p.add_argument("--prediction", type=float, default=0.010, help="OpenVR pose prediction horizon (seconds).")

    # Mapping
    p.add_argument("--calib", type=str, default=str(VR_MODULE_DIR / "vr_joint_calibration.json"), help="Calibration JSON path.")
    p.add_argument("--method", choices=["rbf", "nearest", "linear"], default="rbf", help="Mapping method.")
    p.add_argument("--epsilon", type=float, default=0.0, help="RBF epsilon (0 = auto/median).")
    p.add_argument("--reg", type=float, default=1e-6, help="Regularization strength.")
    p.add_argument("--k", type=int, default=3, help="k for nearest-neighbor.")

    # Runtime reference + smoothing
    p.add_argument("--ref-secs", type=float, default=2.0, help="Seconds to capture runtime RESTING reference.")
    p.add_argument("--smoothing", type=float, default=0.25, help="EMA smoothing alpha (0=no update, 1=no smoothing).")

    # Physics / contact friendliness (matches teleop/gui.py idea)
    p.add_argument("--interp-steps", type=int, default=6, help="Interpolation increments toward targets per VR tick.")
    p.add_argument("--sim-steps-per-interp", type=int, default=2, help="MuJoCo sim steps per interpolation increment.")

    # Gripper mapping
    p.add_argument("--invert-gripper", action="store_true", help="Invert trigger mapping (1=open, 0=close).")

    return p.parse_args()


def capture_reference(
    vr_system: "openvr.IVRSystem",
    left_id: Optional[int],
    right_id: Optional[int],
    secs: float,
    prediction: float,
    poll_interval: float,
) -> Dict[str, HandReference]:
    print("\nHold your RESTING pose now (hands steady). Capturing reference...")
    # Wait/pause until BOTH controllers are actually tracked, then accumulate active seconds.
    active_t = 0.0
    t_wall0 = time.time()
    last_rescan_t = 0.0
    pos_samples: Dict[str, list[np.ndarray]] = {"left": [], "right": []}
    quat_samples: Dict[str, list[np.ndarray]] = {"left": [], "right": []}
    last_len = 0
    while active_t < secs:
        now = time.time()
        if now - last_rescan_t > 1.0 or left_id is None or right_id is None:
            left_id, right_id = find_controllers(vr_system)
            last_rescan_t = now

        if now - t_wall0 > 120.0:
            raise RuntimeError("Controllers were never tracked (pose_valid) during reference capture. Check SteamVR tracking/HMD.")

        poses = vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding,
            prediction,
            openvr.k_unMaxTrackedDeviceCount,
        )
        left_status = "NOT FOUND" if left_id is None else "INIT"
        right_status = "NOT FOUND" if right_id is None else "INIT"
        ok_left = False
        ok_right = False
        if left_id is not None:
            p = poses[left_id]
            if not p.bDeviceIsConnected:
                left_status = "DISCONNECTED"
            elif not p.bPoseIsValid:
                left_status = "NO TRACKING"
            else:
                left_status = "OK"
                ok_left = True
        if right_id is not None:
            p = poses[right_id]
            if not p.bDeviceIsConnected:
                right_status = "DISCONNECTED"
            elif not p.bPoseIsValid:
                right_status = "NO TRACKING"
            else:
                right_status = "OK"
                ok_right = True

        if ok_left and ok_right and left_id is not None and right_id is not None:
            p = poses[left_id]
            pos, quat = openvr_pose_to_pos_quat(p.mDeviceToAbsoluteTracking)
            pos_samples["left"].append(pos)
            quat_samples["left"].append(quat_normalize(quat))

            p = poses[right_id]
            pos, quat = openvr_pose_to_pos_quat(p.mDeviceToAbsoluteTracking)
            pos_samples["right"].append(pos)
            quat_samples["right"].append(quat_normalize(quat))

            active_t += float(max(0.0, poll_interval))

        rem = max(0.0, secs - active_t)
        line = f"  ref remaining: {rem:4.1f}s [L:{left_status} R:{right_status}]"
        last_len = write_inline_status(line, last_len)
        time.sleep(max(0.0, poll_interval))
    sys.stdout.write("\n")

    out: Dict[str, HandReference] = {}
    for hand in ("left", "right"):
        if len(pos_samples[hand]) < 10:
            raise RuntimeError(f"Not enough reference samples for {hand.upper()} (n={len(pos_samples[hand])}).")
        pos_mean = np.mean(np.asarray(pos_samples[hand], dtype=np.float64), axis=0)
        q = np.asarray(quat_samples[hand], dtype=np.float64)
        quat_mean = quat_average(q)
        out[hand] = HandReference(pos_m=pos_mean, quat_wxyz=quat_mean)
    return out


def main() -> None:
    args = parse_args()

    calib_path = Path(args.calib)
    if not calib_path.exists():
        print("[ERROR] Calibration file not found.")
        print(f"  Expected: {calib_path}")
        print("Run: python teleop/vr_module/vr_joint_calibration.py")
        return

    calib = load_calibration(calib_path)
    Xl, Yl, Xr, Yr = build_dataset(calib)

    epsilon = None if float(args.epsilon) <= 0.0 else float(args.epsilon)
    mapper = VRJointMapper(method=args.method, epsilon=epsilon, reg=float(args.reg), k=int(args.k))
    mapper.fit(Xl, Yl, Xr, Yr)
    print(f"[OK] Loaded calibration with {Xl.shape[0]} left poses, {Xr.shape[0]} right poses. Method={args.method}.")

    # ---------------------------------------------------------------------
    # Robosuite env setup (match teleop/gui.py)
    # ---------------------------------------------------------------------
    env = suite.make(
        env_name=str(args.env),
        robots="AeroPiper",
        env_configuration="single-robot",
        has_renderer=not bool(args.no_viewer),
        has_offscreen_renderer=False,
        use_camera_obs=False,
        render_camera=None,
        renderer="mjviewer",
        ignore_done=True,
    )
    env.reset()
    if getattr(env, "viewer", None) is not None:
        try:
            env.viewer.set_camera(camera_id=-1)
        except Exception:
            pass
        try:
            env.viewer.update()
        except Exception:
            pass

    sim = env.sim
    robot = env.robots[0]

    # Arm joint names (robosuite orders: right then left for AeroPiper in this repo)
    arm_joints = list(robot.robot_model.arm_joints)
    arms = getattr(robot, "arms", ["right", "left"])
    split = int(len(arm_joints) / max(1, len(arms)))
    right_arm_joints = arm_joints[:split]
    left_arm_joints = arm_joints[split:]
    if len(left_arm_joints) != 6 or len(right_arm_joints) != 6:
        print(f"[WARN] Expected 6 joints per arm, got L={len(left_arm_joints)} R={len(right_arm_joints)}.")

    # Precompute joint ranges + qvel addrs
    left_lo = np.array([_get_joint_limit(sim, j)[0] for j in left_arm_joints], dtype=np.float64)
    left_hi = np.array([_get_joint_limit(sim, j)[1] for j in left_arm_joints], dtype=np.float64)
    right_lo = np.array([_get_joint_limit(sim, j)[0] for j in right_arm_joints], dtype=np.float64)
    right_hi = np.array([_get_joint_limit(sim, j)[1] for j in right_arm_joints], dtype=np.float64)
    left_qvel_addrs = [sim.model.get_joint_qvel_addr(j) for j in left_arm_joints]
    right_qvel_addrs = [sim.model.get_joint_qvel_addr(j) for j in right_arm_joints]

    # Gripper mapping (match teleop/gui.py)
    left_gripper_joints = list(robot.gripper["left"].joints)
    right_gripper_joints = list(robot.gripper["right"].joints)
    left_gripper_actuators = list(robot.gripper["left"].actuators)
    right_gripper_actuators = list(robot.gripper["right"].actuators)
    left_grip_act_map = _build_gripper_actuator_map(sim, left_gripper_actuators)
    right_grip_act_map = _build_gripper_actuator_map(sim, right_gripper_actuators)
    left_grip_extra_joints = _gripper_extra_flex_joints(left_gripper_joints)
    right_grip_extra_joints = _gripper_extra_flex_joints(right_gripper_joints)
    left_extra_qvel_addrs = {j: sim.model.get_joint_qvel_addr(j) for j in left_grip_extra_joints}
    right_extra_qvel_addrs = {j: sim.model.get_joint_qvel_addr(j) for j in right_grip_extra_joints}

    # ---------------------------------------------------------------------
    # SteamVR init (unchanged)
    # ---------------------------------------------------------------------
    try:
        vr_system = openvr.init(openvr.VRApplication_Other)
    except openvr.OpenVRError as exc:
        print(f"[ERROR] Could not initialize OpenVR: {exc}")
        print("Make sure SteamVR is running and the HMD is awake.")
        env.close()
        return

    try:
        left_id, right_id = find_controllers(vr_system)
        for label, cid in (("LEFT", left_id), ("RIGHT", right_id)):
            if cid is None:
                print(f"[WARN] {label} controller not found at startup.")
            else:
                print(f"[OK] {label} controller detected (device {cid}).")
        _print_controller_debug(vr_system)

        input("\n[READY] Press ENTER when you can see SteamVR tracking and are about to put on the headset...")
        for s in range(20, 0, -1):
            sys.stdout.write(f"\rPut on headset + get into RESTING pose... {s:2d}s ")
            sys.stdout.flush()
            time.sleep(1.0)
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

        refs = capture_reference(
            vr_system,
            left_id,
            right_id,
            secs=float(max(0.5, args.ref_secs)),
            prediction=float(max(0.0, args.prediction)),
            poll_interval=float(max(0.0, args.poll_interval)),
        )
        print("[OK] Runtime reference captured. Starting control loop. Ctrl+C to stop.\n")

        # State for smoothing and hold-last on tracking loss
        cmd_cache: Dict[str, np.ndarray] = {"left": np.zeros(6, dtype=np.float64), "right": np.zeros(6, dtype=np.float64)}
        cache_valid: Dict[str, bool] = {"left": False, "right": False}
        status_cache: Dict[str, str] = {"left": "INIT", "right": "INIT"}
        last_line_len = 0

        prediction = float(max(0.0, args.prediction))
        poll_interval = float(max(0.0, args.poll_interval))
        alpha = float(np.clip(args.smoothing, 0.0, 1.0))
        interp_steps = int(max(1, args.interp_steps))
        sim_steps_per_interp = int(max(0, args.sim_steps_per_interp))
        invert_gripper = bool(args.invert_gripper)
        last_rescan_t = 0.0
        last_debug_t = 0.0

        def _viewer_running() -> bool:
            """
            Robosuite mjviewer renderer stores the underlying MuJoCo viewer at `env.viewer.viewer`.
            That viewer exposes `is_running()`; use it when available so closing the window exits.
            """
            if bool(args.no_viewer):
                return True
            vwrap = getattr(env, "viewer", None)
            v = getattr(vwrap, "viewer", None) if vwrap is not None else None
            if v is None:
                return True
            if hasattr(v, "is_running"):
                try:
                    return bool(v.is_running())
                except Exception:
                    return True
            return True

        while _viewer_running():
            loop_start = time.perf_counter()
            now_wall = time.time()

            if now_wall - last_rescan_t >= CONTROLLER_RESCAN_INTERVAL_S:
                new_left, new_right = find_controllers(vr_system)
                if (new_left, new_right) != (left_id, right_id):
                    left_id, right_id = new_left, new_right
                    print(f"\n[INFO] Controller ids refreshed: LEFT={left_id} RIGHT={right_id}")
                last_rescan_t = now_wall

            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding,
                prediction,
                openvr.k_unMaxTrackedDeviceCount,
            )

            # Per-hand: update cached normalized joint targets (hold-last on tracking loss)
            triggers: Dict[str, float] = {"left": 0.0, "right": 0.0}
            for cid, hand in ((left_id, "left"), (right_id, "right")):
                pos, quat, trigger, status = poll_controller(vr_system, cid, poses)
                triggers[hand] = float(trigger)
                if status_cache.get(hand) != status:
                    print(f"\n[{hand.upper()}] {status}")
                    status_cache[hand] = status

                if status == "OK" and pos is not None and quat is not None:
                    pred = mapper.predict_from_pose(hand, pos, quat, refs[hand])
                    if cache_valid[hand]:
                        cmd_cache[hand] = (1.0 - alpha) * cmd_cache[hand] + alpha * pred
                    else:
                        cmd_cache[hand] = pred
                        cache_valid[hand] = True

            # If either hand isn't OK, periodically print OpenVR's view of devices.
            if (status_cache.get("left") != "OK" or status_cache.get("right") != "OK") and (
                now_wall - last_debug_t >= OPENVR_DEBUG_PRINT_INTERVAL_S
            ):
                _print_controller_debug(vr_system)
                last_debug_t = now_wall

            # Convert normalized [-1,1] targets into joint qpos targets.
            left_tgt = _normalized_to_joint_range(cmd_cache["left"], left_lo, left_hi)
            right_tgt = _normalized_to_joint_range(cmd_cache["right"], right_lo, right_hi)

            # Current qpos
            left_curr = np.array([float(sim.data.get_joint_qpos(j)) for j in left_arm_joints], dtype=np.float64)
            right_curr = np.array([float(sim.data.get_joint_qpos(j)) for j in right_arm_joints], dtype=np.float64)

            # Gripper alpha from trigger (match teleop/gui.py: 0=open, 1=close)
            l_alpha = float(np.clip(1.0 - triggers["left"] if invert_gripper else triggers["left"], 0.0, 1.0))
            r_alpha = float(np.clip(1.0 - triggers["right"] if invert_gripper else triggers["right"], 0.0, 1.0))

            # Interpolate toward targets + step physics so contacts get resolved.
            for k in range(interp_steps):
                a = float(k + 1) / float(interp_steps)
                left_next = left_curr + a * (left_tgt - left_curr)
                right_next = right_curr + a * (right_tgt - right_curr)

                for j, q, addr in zip(left_arm_joints, left_next, left_qvel_addrs):
                    sim.data.set_joint_qpos(j, float(q))
                    _set_joint_qvel_zero_by_addr(sim, addr)
                for j, q, addr in zip(right_arm_joints, right_next, right_qvel_addrs):
                    sim.data.set_joint_qpos(j, float(q))
                    _set_joint_qvel_zero_by_addr(sim, addr)

                # Grippers: actuators + extra joints (same as teleop/gui.py intent)
                if left_grip_act_map:
                    for aid, lo, hi in left_grip_act_map["controlled"]:
                        sim.data.ctrl[aid] = lo + l_alpha * (hi - lo)
                    if left_grip_act_map["thumb_abd"] is not None:
                        aid, lo, hi = left_grip_act_map["thumb_abd"]
                        sim.data.ctrl[aid] = hi
                if right_grip_act_map:
                    for aid, lo, hi in right_grip_act_map["controlled"]:
                        sim.data.ctrl[aid] = lo + r_alpha * (hi - lo)
                    if right_grip_act_map["thumb_abd"] is not None:
                        aid, lo, hi = right_grip_act_map["thumb_abd"]
                        sim.data.ctrl[aid] = hi

                for j in left_grip_extra_joints:
                    lo, hi, _ = _get_joint_limit(sim, j)
                    sim.data.set_joint_qpos(j, lo + l_alpha * (hi - lo))
                    _set_joint_qvel_zero_by_addr(sim, left_extra_qvel_addrs[j])
                for j in right_grip_extra_joints:
                    lo, hi, _ = _get_joint_limit(sim, j)
                    sim.data.set_joint_qpos(j, lo + r_alpha * (hi - lo))
                    _set_joint_qvel_zero_by_addr(sim, right_extra_qvel_addrs[j])

                sim.forward()
                for _ in range(sim_steps_per_interp):
                    sim.step()

            # Keep Robosuite mjviewer synced
            if getattr(env, "viewer", None) is not None and hasattr(env.viewer, "update"):
                try:
                    env.viewer.update()
                except Exception:
                    pass

            # Inline status
            l = cmd_cache["left"]
            r = cmd_cache["right"]
            line = (
                f"L:[{l[0]:+.2f},{l[1]:+.2f},{l[2]:+.2f},{l[3]:+.2f},{l[4]:+.2f},{l[5]:+.2f}] {status_cache.get('left','NA')} | "
                f"R:[{r[0]:+.2f},{r[1]:+.2f},{r[2]:+.2f},{r[3]:+.2f},{r[4]:+.2f},{r[5]:+.2f}] {status_cache.get('right','NA')} | "
                f"Grip L:{l_alpha:.2f} R:{r_alpha:.2f}"
            )
            last_line_len = write_inline_status(line, last_line_len)

            elapsed = time.perf_counter() - loop_start
            if poll_interval > 0 and elapsed < poll_interval:
                time.sleep(poll_interval - elapsed)

    except KeyboardInterrupt:
        sys.stdout.write("\n")
        print("\n[INFO] Interrupted.")
    finally:
        try:
            openvr.shutdown()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        print("SteamVR connection closed.")


if __name__ == "__main__":
    main()


