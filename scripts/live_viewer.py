"""Simple MuJoCo viewer that auto-reloads when the MJCF file changes."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer as mj_viewer


MIDPOSE = {
    "joint1": 0.0,
    "joint2": 1.57,
    "joint3": -1.35,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "left_joint1": 0.0,
    "left_joint2": 1.57,
    "left_joint3": -1.35,
    "left_joint4": 0.0,
    "left_joint5": 0.0,
    "left_joint6": 0.0,
}


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Launch MuJoCo viewer and hot-reload when the MJCF changes.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
      "--mjcf",
      default="assets/scene.xml",
      help="Path to the MJCF file to watch.",
  )
  parser.add_argument(
      "--interval",
      type=float,
      default=0.25,
      help="Polling interval (seconds) for checking file changes.",
  )
  return parser.parse_args()


def _apply_midpose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  """Set specified joints to desired mid values; zero everything else."""
  data.qpos[:] = 0
  data.qvel[:] = 0
  data.ctrl[:] = 0

  for jname, val in MIDPOSE.items():
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
      continue
    qadr = model.jnt_qposadr[jid]
    data.qpos[qadr] = val

    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
    if act_id >= 0:
      data.ctrl[act_id] = val


def _load_model(xml_path: Path):
  """Returns (loader, model, data) for the given MJCF path."""
  loader = mj_viewer._file_loader(str(xml_path))  # pylint: disable=protected-access
  model, data, _ = loader()
  _apply_midpose(model, data)
  return loader, model, data

def main() -> None:
  args = _parse_args()
  xml_path = Path(args.mjcf).expanduser().resolve()
  if not xml_path.exists():
    print(f"[error] MJCF file not found: {xml_path}", file=sys.stderr)
    sys.exit(1)

  loader, model, data = _load_model(xml_path)
  last_mtime = xml_path.stat().st_mtime

  print(f"Watching {xml_path}")
  print("Edit and save the file to trigger an in-place reload. Press Ctrl+C to exit.")

  with mj_viewer.launch_passive(model, data) as handle:
    sim = handle._get_sim()  # pylint: disable=protected-access
    if sim is None:
      print("[error] Failed to start MuJoCo viewer.", file=sys.stderr)
      sys.exit(1)

    def _on_reload_success():
      ts = time.strftime("%H:%M:%S")
      print(f"[{ts}] Reloaded {xml_path.name}")
      sim = handle._get_sim()  # pylint: disable=protected-access
      if sim is not None:
        _apply_midpose(sim.model, sim.data)

    try:
      while handle.is_running():
        handle.sync()
        time.sleep(max(args.interval, 0.01))

        try:
          current_mtime = xml_path.stat().st_mtime
        except FileNotFoundError:
          continue

        if current_mtime == last_mtime:
          continue
        last_mtime = current_mtime

        sim = handle._get_sim()  # pylint: disable=protected-access
        if sim is None:
          break
        try:
          mj_viewer._reload(  # pylint: disable=protected-access
              sim, loader, notify_loaded=_on_reload_success
          )
        except Exception as exc:  # pylint: disable=broad-except
          print(f"[reload failed] {exc}", file=sys.stderr)
    except KeyboardInterrupt:
      pass
    finally:
      handle.close()


if __name__ == "__main__":
  main()

