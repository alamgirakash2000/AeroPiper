"""
VR -> Robot Joint Mapping (calibrated, nonlinear).

This module supports the calibration workflow described by the user:
  - Record stable VR controller poses for a handful of named "anchor" poses
  - Associate each anchor with desired robot joint targets (normalized [-1,1])
  - Fit a smooth nonlinear interpolator (default: Gaussian RBF) per hand

Runtime mapping uses *relative* controller motion:
  feature = [dpos/POS_RANGE, rotvec(dq)/pi]
where dq is the orientation delta from a runtime "resting" reference pose.

This makes the calibration portable across sessions even if SteamVR's world
origin shifts, as long as the user re-captures the same resting pose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import numpy as np

Hand = Literal["left", "right"]
Method = Literal["rbf", "nearest", "linear"]

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_FILE = MODULE_DIR / "vr_joint_calibration.json"

POS_RANGE_DEFAULT = 1.0
ROT_RANGE_DEFAULT = float(np.pi)


def clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q = q / n
    # Keep sign consistent (q and -q represent the same rotation)
    if q[0] < 0.0:
        q = -q
    return q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion product a ⊗ b (w,x,y,z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q
    return np.array([w, -x, -y, -z], dtype=np.float64) / max(1e-12, float(np.dot(q, q)))


def quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """
    Quaternion -> rotation vector (axis * angle) in radians.
    Assumes q is a relative rotation (ideally close to identity).
    """
    q = quat_normalize(q)
    w, x, y, z = q
    v = np.array([x, y, z], dtype=np.float64)
    nv = float(np.linalg.norm(v))
    if nv < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(nv, w)
    axis = v / nv
    # Wrap to [-pi, pi] for stability
    if angle > np.pi:
        angle = angle - 2.0 * np.pi
    return axis * angle


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w,x,y,z)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = float(np.trace(R))
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def quat_average(quats: np.ndarray) -> np.ndarray:
    """
    Average quaternions (wxyz) using the Markley method.
    quats: (N,4)
    """
    Q = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
    if Q.shape[0] == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    Q = np.array([quat_normalize(q) for q in Q], dtype=np.float64)
    A = np.zeros((4, 4), dtype=np.float64)
    for q in Q:
        A += np.outer(q, q)
    A /= float(Q.shape[0])
    eigvals, eigvecs = np.linalg.eigh(A)
    q_mean = eigvecs[:, int(np.argmax(eigvals))]
    return quat_normalize(q_mean)


def feature_from_pose(
    pos_m: np.ndarray,
    quat_wxyz: np.ndarray,
    ref_pos_m: np.ndarray,
    ref_quat_wxyz: np.ndarray,
    pos_range: float = POS_RANGE_DEFAULT,
    rot_range: float = ROT_RANGE_DEFAULT,
) -> np.ndarray:
    """Return 6D feature: [dpos/pos_range, rotvec(dq)/rot_range]."""
    pos_m = np.asarray(pos_m, dtype=np.float64).reshape(3)
    quat_wxyz = quat_normalize(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))
    ref_pos_m = np.asarray(ref_pos_m, dtype=np.float64).reshape(3)
    ref_quat_wxyz = quat_normalize(np.asarray(ref_quat_wxyz, dtype=np.float64).reshape(4))

    dpos = (pos_m - ref_pos_m) / float(max(1e-9, pos_range))
    dq = quat_mul(quat_inv(ref_quat_wxyz), quat_wxyz)
    drot = quat_to_rotvec(dq) / float(max(1e-9, rot_range))
    x = np.concatenate([dpos, drot]).astype(np.float64)
    return np.clip(x, -2.0, 2.0)  # allow a bit beyond [-1,1] without exploding


def pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    # (N,1,D) - (1,N,D) -> (N,N,D)
    diffs = X[:, None, :] - X[None, :, :]
    return np.sum(diffs * diffs, axis=-1)


class RBFRegressor:
    """Gaussian RBF regressor with Tikhonov regularization."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, epsilon: Optional[float] = None, reg: float = 1e-6) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.Y = np.asarray(Y, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.Y.ndim != 2:
            raise ValueError("Y must be 2D")
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("X and Y must have same N")

        self.N = self.X.shape[0]
        self.D = self.X.shape[1]
        self.M = self.Y.shape[1]

        d2 = pairwise_sq_dists(self.X)
        # Choose epsilon as median pairwise distance (excluding zeros)
        if epsilon is None:
            off = d2[np.triu_indices(self.N, k=1)]
            med = float(np.median(off)) if off.size else 1.0
            eps = float(np.sqrt(max(med, 1e-6)))
        else:
            eps = float(max(1e-6, epsilon))
        self.epsilon = eps
        self.reg = float(max(0.0, reg))

        K = np.exp(-d2 / (2.0 * eps * eps))
        K = K + self.reg * np.eye(self.N, dtype=np.float64)
        self.W = np.linalg.solve(K, self.Y)  # (N,M)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != self.D:
            raise ValueError(f"Expected feature dim {self.D}, got {x.shape[1]}")
        diffs = self.X - x  # (N,D)
        d2 = np.sum(diffs * diffs, axis=1)  # (N,)
        k = np.exp(-d2 / (2.0 * self.epsilon * self.epsilon))  # (N,)
        y = k @ self.W  # (M,)
        return y.astype(np.float64)


class NearestRegressor:
    """k-NN inverse-distance weighted regressor."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, k: int = 3, eps: float = 1e-3) -> None:
        self.X = np.asarray(X, dtype=np.float64)
        self.Y = np.asarray(Y, dtype=np.float64)
        self.k = int(max(1, min(k, self.X.shape[0])))
        self.eps = float(max(1e-9, eps))

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        diffs = self.X - x
        d = np.linalg.norm(diffs, axis=1)
        idx = np.argsort(d)[: self.k]
        dd = d[idx]
        w = 1.0 / (dd + self.eps)
        w = w / max(1e-12, float(np.sum(w)))
        y = (w[:, None] * self.Y[idx]).sum(axis=0)
        return y.astype(np.float64)


class LinearRegressor:
    """Per-output linear regression with bias."""

    def __init__(self, X: np.ndarray, Y: np.ndarray, reg: float = 1e-6) -> None:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        Xb = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
        # Ridge regression closed-form: (X^T X + λI)^-1 X^T Y
        XtX = Xb.T @ Xb
        lam = float(max(0.0, reg))
        XtX = XtX + lam * np.eye(XtX.shape[0], dtype=np.float64)
        self.B = np.linalg.solve(XtX, Xb.T @ Y)  # (D+1,M)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        xb = np.concatenate([x, [1.0]])
        return (xb @ self.B).astype(np.float64)


@dataclass
class HandReference:
    pos_m: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)


@dataclass
class VRJointMapper:
    method: Method = "rbf"
    epsilon: Optional[float] = None
    reg: float = 1e-6
    k: int = 3
    pos_range: float = POS_RANGE_DEFAULT
    rot_range: float = ROT_RANGE_DEFAULT

    _models: Dict[Hand, object] = None  # type: ignore[assignment]

    def fit(self, X_left: np.ndarray, Y_left: np.ndarray, X_right: np.ndarray, Y_right: np.ndarray) -> None:
        self._models = {}
        self._models["left"] = self._fit_one(X_left, Y_left)
        self._models["right"] = self._fit_one(X_right, Y_right)

    def _fit_one(self, X: np.ndarray, Y: np.ndarray) -> object:
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.shape[0] < 2:
            # Degenerate: just return a nearest regressor that always returns the single point.
            return NearestRegressor(X, Y, k=1)
        if self.method == "nearest":
            return NearestRegressor(X, Y, k=self.k)
        if self.method == "linear":
            return LinearRegressor(X, Y, reg=self.reg)
        return RBFRegressor(X, Y, epsilon=self.epsilon, reg=self.reg)

    def predict_from_feature(self, hand: Hand, x: np.ndarray) -> np.ndarray:
        if not self._models or hand not in self._models:
            raise RuntimeError("Model is not fit/loaded.")
        model = self._models[hand]
        y = model.predict(x)  # type: ignore[attr-defined]
        return np.clip(y, -1.0, 1.0)

    def predict_from_pose(self, hand: Hand, pos_m: np.ndarray, quat_wxyz: np.ndarray, ref: HandReference) -> np.ndarray:
        x = feature_from_pose(pos_m, quat_wxyz, ref.pos_m, ref.quat_wxyz, self.pos_range, self.rot_range)
        return self.predict_from_feature(hand, x)


def load_calibration(path: str | Path = DEFAULT_CALIBRATION_FILE) -> dict:
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(calib: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_left, Y_left, X_right, Y_right) from calibration JSON."""
    poses = calib.get("poses", [])
    X_left, Y_left = [], []
    X_right, Y_right = [], []

    for pose in poses:
        name = pose.get("name", "unknown")
        features = pose.get("features", {})
        targets = pose.get("robot_targets_norm", {})
        for hand in ("left", "right"):
            if hand not in features or hand not in targets:
                continue
            x = np.asarray(features[hand], dtype=np.float64).reshape(6)
            y = np.asarray(targets[hand], dtype=np.float64).reshape(6)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
                continue
            if hand == "left":
                X_left.append(x)
                Y_left.append(y)
            else:
                X_right.append(x)
                Y_right.append(y)

        # Ensure at least resting exists in the file; otherwise warn upstream.
        if name == "resting":
            pass

    if not X_left or not X_right:
        raise ValueError("Calibration dataset incomplete: need at least 1 pose per hand.")

    return (
        np.vstack(X_left).astype(np.float64),
        np.vstack(Y_left).astype(np.float64),
        np.vstack(X_right).astype(np.float64),
        np.vstack(Y_right).astype(np.float64),
    )


