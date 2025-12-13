#!/usr/bin/env python3
"""
Manual arm control for AeroPiper pick-place (real-time distances).

Usage examples:
  conda run -n aeropiper_mjlab python scripts/manual_control.py --render
  conda run -n aeropiper_mjlab python scripts/manual_control.py --render --arm left

Controls:
  - On each prompt, enter 6 numbers: arm_flag j1 j2 j3 j4 j5
    * arm_flag >= 0 => right arm, < 0 => left arm
    * j1..j5 are deltas in normalized [-1, 1]; joint6 is held at 0
  - Press Enter with no input to send zeros (hold pose)
  - Type 'q' to quit

Displays per-step distances:
  - cube->target, cube->EE(active), EE(active)->target
  - per-arm EE->target (right/left)
"""

from __future__ import annotations

import argparse
import sys
from typing import List

import numpy as np

from envs.wrappers.pick_place_auto_grasp import AeroPiperPickPlaceAutoGrasp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual control with distance readout")
    parser.add_argument("--render", action="store_true", help="Open Mujoco viewer")
    parser.add_argument("--arm", choices=["right", "left"], default="right", help="Initial arm flag")
    parser.add_argument("--seed", type=int, default=0, help="Seed for env reset")
    return parser.parse_args()


def parse_action(line: str, default_arm: str) -> np.ndarray:
    """
    Parse a line of input into a 6-D action.
    Format: arm_flag j1 j2 j3 j4 j5
    """
    line = line.strip()
    if not line:
        return np.zeros(6, dtype=np.float64)
    if line.lower() == "q":
        raise KeyboardInterrupt

    parts: List[float] = [float(x) for x in line.split()]
    if len(parts) != 6:
        raise ValueError("Expect 6 numbers: arm_flag j1 j2 j3 j4 j5")
    action = np.asarray(parts, dtype=np.float64)
    action = np.clip(action, -1.0, 1.0)
    return action


def main() -> None:
    args = parse_args()
    render_mode = "human" if args.render else None

    env = AeroPiperPickPlaceAutoGrasp(render_mode=render_mode, randomize_objects=False)
    obs, info = env.reset(seed=args.seed)

    # Set initial arm
    init_flag = 1.0 if args.arm == "right" else -1.0
    action = np.zeros(6, dtype=np.float64)
    action[0] = init_flag

    print("Manual control started. Enter: arm_flag j1 j2 j3 j4 j5 (all in [-1,1])")
    print("Press Enter to hold. Type 'q' to quit.\n")

    step = 0
    try:
        while True:
            # Display distances
            cube = info["cube_pos"]
            target = info["target_pos"]
            active_arm = info["active_arm"]
            dist_cube_target = info["dist_cube_target"]
            dist_cube_hand = info["dist_cube_hand"]
            dist_ee_target = info.get("dist_ee_target", np.nan)
            dist_ee_target_r = info.get("dist_ee_target_right", np.nan)
            dist_ee_target_l = info.get("dist_ee_target_left", np.nan)

            print(
                f"[{step:04d}] arm={active_arm:5s} "
                f"cube→target={dist_cube_target: .3f} "
                f"cube→EE={dist_cube_hand: .3f} "
                f"EE→target={dist_ee_target: .3f} "
                f"EE→target(R)={dist_ee_target_r: .3f} "
                f"EE→target(L)={dist_ee_target_l: .3f}"
            )

            # Get user action
            line = input("action> ")
            try:
                action = parse_action(line, args.arm)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001
                print(f"Invalid input: {exc}")
                continue

            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if terminated or truncated:
                print("Episode ended, resetting.")
                obs, info = env.reset(seed=args.seed)
                step = 0

    except KeyboardInterrupt:
        print("\nExiting manual control.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
