#!/usr/bin/env python3
"""
Control the physical Aero hand with real-time webcam gestures.

This script mirrors `left_hand_gesture.py` but connects to the actual Aero hand
hardware instead of MuJoCo. Only the hand is commanded; the arm remains
stationary. The script continuously tracks a real left hand via MediaPipe,
filters the seven AeroPiper physical DOFs (thumb abduction + 6 flexions), and
streams them to the robot through the Aero Open SDK.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

SCRIPT_DIR = os.path.dirname(__file__)
MODULE_DIR = os.path.join(SCRIPT_DIR, "module")
if MODULE_DIR not in sys.path:
    sys.path.append(MODULE_DIR)

from hand_tracker import HandGestureController  # type: ignore[import]

try:
    from aero_open_sdk.aero_hand import AeroHand  # type: ignore[import]
except ImportError as exc:  # pragma: no cover - hardware dependency
    raise ImportError(
        "AeroHand from `aero_open_sdk` is required for physical control.\n"
        "Install the Aero Open SDK inside the `aeropiper` environment."
    ) from exc

HAND_LABELS = ["ThumbAbd", "Thumb1", "Thumb2", "Index", "Middle", "Ring", "Pinky"]

DEFAULT_PORT = os.environ.get(
    "AEROPIPER_HAND_PORT",
    "COM5" if os.name == "nt" else "/dev/ttyACM0",
)
DEFAULT_BAUDRATE = int(os.environ.get("AEROPIPER_HAND_BAUD", "921600"))


def _apply_deadband(values: np.ndarray, previous: np.ndarray | None, threshold: float):
    current = np.asarray(values, dtype=float)
    if previous is None:
        return current.copy()
    filtered = previous.copy()
    mask = np.abs(current - previous) >= threshold
    filtered[mask] = current[mask]
    return filtered


def _format_hand_stats(values: np.ndarray) -> str:
    return " ".join(f"{name}:{val:5.1f}" for name, val in zip(HAND_LABELS, values))


@dataclass
class HandStatus:
    values: np.ndarray
    timestamp: float


class AeroHandCommandSender:
    """Thin wrapper that converts 7-DOF commands to the 16 joint commands AeroHand expects."""

    def __init__(self, port: str, baudrate: int, dry_run: bool):
        self._port = port
        self._baudrate = baudrate
        self._dry_run = dry_run
        self._hand: Optional[AeroHand] = None
        self._last_status = HandStatus(np.zeros(7, dtype=float), 0.0)

        if self._dry_run:
            print("[DRY-RUN] Skipping AeroHand connection – values will only be printed.")
            return

        print(f"→ Connecting to Aero hand on {port} @ {baudrate} baud ...")
        self._hand = AeroHand(port=port, baudrate=baudrate)
        time.sleep(0.5)
        print("    ✓ Connected to Aero hand")

    @property
    def port(self) -> str:
        return self._port

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    def send(self, values: np.ndarray):
        clipped = np.clip(np.asarray(values, dtype=float), 0.0, 100.0)
        self._last_status = HandStatus(clipped, time.time())

        if self._dry_run:
            return

        assert self._hand is not None  # For type-checkers
        gesture7 = clipped.tolist()
        gesture16 = self._hand.convert_seven_joints_to_sixteen(gesture7)
        self._hand.set_joint_positions(gesture16)

    def last_update_age(self) -> Optional[float]:
        if self._last_status.timestamp == 0.0:
            return None
        return max(time.time() - self._last_status.timestamp, 0.0)

    def last_values(self) -> np.ndarray:
        return self._last_status.values

    def open_hand(self):
        self.send(np.zeros(7, dtype=float))

    def close(self):
        if self._hand is not None:
            try:
                self._hand.close()
            except Exception as exc:  # pragma: no cover - best effort cleanup
                print(f"[WARN] Failed to close Aero hand cleanly: {exc}")


def _print_status(sender: AeroHandCommandSender, prefix: str = "[Hand]"):
    values = sender.last_values()
    stats = _format_hand_stats(values)
    age = sender.last_update_age()

    if sender.dry_run:
        transport = "DRY-RUN"
    else:
        transport = sender.port

    if age is None:
        metadata = f"{transport}: waiting for first command"
    else:
        metadata = f"{transport}: sent {age * 1000.0:4.0f} ms ago"

    print(f"\r{prefix} {stats} | {metadata}", end="", flush=True)


def run_physical_control(
    controller: HandGestureController,
    sender: AeroHandCommandSender,
    update_threshold: float,
    print_interval: float,
):
    last_sent: Optional[np.ndarray] = None
    last_print = 0.0

    try:
        while not controller.should_stop():
            physical = controller.update()
            filtered = _apply_deadband(physical, last_sent, update_threshold)

            needs_send = last_sent is None or not np.array_equal(filtered, last_sent)
            if needs_send:
                sender.send(filtered)
                last_sent = filtered

            now = time.time()
            if now - last_print >= max(print_interval, 1e-3):
                _print_status(sender)
                last_print = now
    finally:
        print()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control the physical Aero hand with real-time webcam gestures."
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam index to open (default: 0).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the OpenCV preview window if running headless.",
    )
    parser.add_argument(
        "--no-mirror-preview",
        action="store_true",
        help="Disable mirroring in the preview window.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.35,
        help="EMA weight in [0,1] for new gesture samples (default: 0.35).",
    )
    parser.add_argument(
        "--update-threshold",
        type=float,
        default=5.0,
        help="Minimum change (0-100 scale) required to emit a new value.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.1,
        help="Seconds between CLI print updates.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default=DEFAULT_PORT,
        help=f"Serial port for the Aero hand (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=DEFAULT_BAUDRATE,
        help=f"Serial baudrate for the Aero hand (default: {DEFAULT_BAUDRATE}).",
    )
    parser.add_argument(
        "--no-auto-open",
        action="store_true",
        help="Skip sending an open-hand command on startup and shutdown.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the tracker without connecting to the physical hand.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    controller: Optional[HandGestureController] = None
    sender: Optional[AeroHandCommandSender] = None

    try:
        controller = HandGestureController(
            camera_index=args.camera_index,
            show_preview=not args.no_preview,
            mirror_preview=not args.no_mirror_preview,
            smoothing=args.smoothing,
        )

        sender = AeroHandCommandSender(args.port, args.baudrate, args.dry_run)

        if not args.no_auto_open:
            print("→ Sending open-hand pose for safety...")
            sender.open_hand()

        threshold = max(args.update_threshold, 0.0)
        run_physical_control(controller, sender, threshold, args.print_interval)

    except KeyboardInterrupt:
        print("\nInterrupted, shutting down.")
    finally:
        if controller is not None:
            controller.close()
        if sender is not None:
            if not args.no_auto_open:
                try:
                    print("\n→ Returning hand to open pose...")
                    sender.open_hand()
                except Exception as exc:  # pragma: no cover - best effort
                    print(f"[WARN] Failed to re-open hand: {exc}")
            sender.close()


if __name__ == "__main__":
    main()


