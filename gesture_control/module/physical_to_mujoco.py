import numpy as np
from typing import Iterable

# Per-DOF MuJoCo control ranges (min, max) in the physical/MuJoCo order:
# [thumb_abd, thumb_flex (th1), thumb_tendon (th2), index, middle, ring, pinky]
MJ_MIN = np.array([
    -0.1,      # thumb_abd
    0.026152,  # thumb_flex (thumb1)
    0.081568,  # thumb_tendon (thumb2)
    0.05852,   # index
    0.05852,   # middle
    0.05852,   # ring
    0.05852,   # pinky
], dtype=float)

MJ_MAX = np.array([
    1.75,      # thumb_abd
    0.038389,  # thumb_flex (thumb1)
    0.112138,  # thumb_tendon (thumb2)
    0.110387,  # index
    0.110387,  # middle
    0.110387,  # ring
    0.110387,  # pinky
], dtype=float)

# Direction mask: True means the physical value is inverted when mapping to MuJoCo
# (0=open -> max control, 100=closed -> min control).
# Thumb abduction (DOF 0) is non-inverted; the rest are inverted.
INVERTED = np.array([False, True, True, True, True, True, True], dtype=bool)

def _normalize_physical(values: np.ndarray) -> np.ndarray:
    """Accept values in 0..100 or 0..1, return normalized 0..1."""
    scale = 100.0 if values.max(initial=0.0) > 1.0 or values.min(initial=0.0) < 0.0 else 1.0
    values = values / scale
    values = np.nan_to_num(values, nan=0.0)
    return np.clip(values, 0.0, 1.0)

def _normalize_mujoco(values: np.ndarray) -> np.ndarray:
    """Normalize MuJoCo values to 0..1 relative to MJ_MIN..MJ_MAX per-DOF."""
    span = MJ_MAX - MJ_MIN
    # Avoid divide-by-zero if any span were zero (not expected)
    span = np.where(span == 0.0, 1.0, span)
    return np.clip((values - MJ_MIN) / span, 0.0, 1.0)

def physical_to_mujoco(physical_values: Iterable[float]) -> np.ndarray:
    """
    Convert physical values [thumb_abd, thumb_flex, thumb_tendon, index, middle, ring, pinky]
    to MuJoCo control values using the same direction as the physical robot:
      - 0  -> fingers/thumb fully open/extended
      - 90/100 -> fingers/thumb fully closed/flexed
    Accepts either 0..100 (preferred) or already normalized 0..1 inputs.
    """
    values = np.asarray(list(physical_values), dtype=float)
    normalized = _normalize_physical(values)
    span = MJ_MAX - MJ_MIN

    out = np.empty_like(normalized)
    inverted = INVERTED

    # Inverted DOFs: 0 -> MJ_MAX, 100 -> MJ_MIN
    out[inverted] = MJ_MAX[inverted] - span[inverted] * normalized[inverted]
    # Non-inverted DOFs (none currently, but keep for completeness)
    non_inverted = ~inverted
    if np.any(non_inverted):
        out[non_inverted] = MJ_MIN[non_inverted] + span[non_inverted] * normalized[non_inverted]

    return out

def mujoco_to_physical(mujoco_values: Iterable[float]) -> np.ndarray:
    """
    Convert MuJoCo control values back to physical values in 0..100 scale following
    the same direction convention as `physical_to_mujoco`.
    """
    values = np.asarray(list(mujoco_values), dtype=float)
    norm = _normalize_mujoco(values)

    physical = np.empty_like(norm)
    inverted = INVERTED
    physical[inverted] = (1.0 - norm[inverted]) * 100.0
    non_inverted = ~inverted
    if np.any(non_inverted):
        physical[non_inverted] = norm[non_inverted] * 100.0

    return np.clip(physical, 0.0, 100.0)


