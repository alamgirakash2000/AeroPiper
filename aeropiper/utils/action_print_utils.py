import numpy as np


def print_robot_action_bounds(env) -> None:
    """
    Prints the action dimension names (in-order) and their min/max bounds for env.step(action).

    This uses the *current* controller action ordering (robot._action_split_indexes), so if the
    robot/controller changes ordering, this printout matches what env.step() expects.
    """
    low, high = env.action_spec

    # Support multi-robot envs (env.action_spec is concatenated across robots)
    action_offset = 0
    for robot_idx, robot in enumerate(env.robots):
        # Defensive: ensure we have the split info (set during controller init)
        split = getattr(robot, "_action_split_indexes", None) or getattr(robot, "composite_controller", None)._action_split_indexes

        # Helper to create per-dimension names for arms / grippers
        def dim_names_for_part(part_name: str, dim: int):
            # Arms
            if part_name in getattr(robot, "arms", []):
                ctrl_type = None
                try:
                    ctrl_type = robot.part_controller_config[part_name]["type"]
                except Exception:
                    ctrl_type = None

                # Common OSC naming
                if dim == 6 and (ctrl_type or "").startswith("OSC"):
                    return [
                        f"{part_name}_dpos_x",
                        f"{part_name}_dpos_y",
                        f"{part_name}_dpos_z",
                        f"{part_name}_drot_x",
                        f"{part_name}_drot_y",
                        f"{part_name}_drot_z",
                    ]
                if dim == 3 and (ctrl_type or "").startswith("OSC"):
                    return [f"{part_name}_dpos_x", f"{part_name}_dpos_y", f"{part_name}_dpos_z"]

                # Joint controllers: label with actual joint names if available
                if (ctrl_type or "").startswith("JOINT") and hasattr(robot, "robot_model") and hasattr(robot.robot_model, "arm_joints"):
                    try:
                        split_idx = robot._joint_split_idx
                        arm_joints = list(robot.robot_model.arm_joints)
                        jnames = arm_joints[:split_idx] if part_name == "right" else arm_joints[split_idx:]
                        if len(jnames) == dim:
                            return [f"{part_name}_{jn}" for jn in jnames]
                    except Exception:
                        pass

                return [f"{part_name}_{i}" for i in range(dim)]

            # Grippers
            if part_name.endswith("_gripper"):
                arm = part_name.replace("_gripper", "")
                try:
                    gj = robot.gripper_joints.get(arm, None)
                    if gj is not None and len(gj) == dim:
                        return [f"{part_name}_{jn}" for jn in gj]
                except Exception:
                    pass
                return [f"{part_name}_{i}" for i in range(dim)]

            # Fallback
            return [f"{part_name}_{i}" for i in range(dim)]

        for part_name, (start, end) in split.items():
            dim = end - start
            names = dim_names_for_part(part_name, dim)
            for local_i in range(dim):
                global_i = action_offset + start + local_i
                print(f"{global_i:03d} {names[local_i]} {float(low[global_i]): .6f} {float(high[global_i]): .6f}")

        action_offset += robot.action_dim


