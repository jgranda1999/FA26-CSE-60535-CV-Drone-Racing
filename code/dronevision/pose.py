"""
Stage 4 - gate pose by PnP.

The gate's inner square is a planar object of known metric size (1.5 m side,
simulator spec 3.7). Four image corners of a known planar square are exactly
what `SOLVEPNP_IPPE_SQUARE` is designed for, so a single detection recovers the
gate's position and orientation in the camera frame.

Object points are defined in a gate-centred frame with the square in the z = 0
plane, ordered TL, TR, BR, BL to match `detector.order_quad`:

        (-s/2, +s/2, 0)   (+s/2, +s/2, 0)
        (-s/2, -s/2, 0)   (+s/2, -s/2, 0)

From the resulting translation `tvec` we report:
  * range      = ||tvec||                          (metres to the gate centre)
  * bearing    = atan2(x, z)                        (left/right, +right, radians)
  * elevation  = atan2(-y, z)                       (up/down, +up, radians)

Because the logs carry no pose ground truth yet (Part 2), these are pose
*pseudo-estimates*; their correctness is checked indirectly in Part 3 by the
range-vs-time curve.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import CameraModel

GATE_INNER_SIZE_M = 1.5   # simulator spec 3.7: inner square 1500 mm


@dataclass
class GatePose:
    rvec: np.ndarray         # (3, 1) Rodrigues rotation, gate -> camera
    tvec: np.ndarray         # (3, 1) translation, gate centre in camera frame (m)
    range_m: float
    bearing_rad: float       # + is to the right of the optical axis
    elevation_rad: float     # + is above the optical axis
    reproj_error_px: float


def _object_points(size_m: float) -> np.ndarray:
    s = size_m / 2.0
    return np.array(
        [[-s, +s, 0.0],   # TL
         [+s, +s, 0.0],   # TR
         [+s, -s, 0.0],   # BR
         [-s, -s, 0.0]],  # BL
        dtype=np.float64,
    )


def estimate_gate_pose(
    corners: np.ndarray,
    camera: CameraModel | None = None,
    size_m: float = GATE_INNER_SIZE_M,
) -> GatePose | None:
    """Estimate the pose of one gate from its four ordered inner corners."""
    camera = camera or CameraModel()
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    object_points = _object_points(size_m)

    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera.K,
        camera.dist,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ok:
        return None

    t = tvec.ravel()
    x, y, z = float(t[0]), float(t[1]), float(t[2])
    range_m = float(np.linalg.norm(t))
    bearing = float(np.arctan2(x, z))
    elevation = float(np.arctan2(-y, z))

    proj, _ = cv2.projectPoints(object_points, rvec, tvec, camera.K, camera.dist)
    reproj_error = float(np.linalg.norm(proj.reshape(4, 2) - image_points, axis=1).mean())

    return GatePose(
        rvec=rvec,
        tvec=tvec,
        range_m=range_m,
        bearing_rad=bearing,
        elevation_rad=elevation,
        reproj_error_px=reproj_error,
    )
