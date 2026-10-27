"""
Stage 5 - visualization helpers for the Part 3 report figures.

All drawing is done with OpenCV so the only hard dependency is opencv + numpy.
`build_pipeline_panel` is used for the stage-by-stage figure; `annotate_frame`
produces the per-frame overlay used in the detection gallery.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detector import GateDetection
from .pose import GatePose
from .raceline import Raceline

GREEN = (0, 255, 0)
AMBER = (0, 180, 255)
MAGENTA = (255, 0, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)


def _put(img, text, org, scale=0.5, color=WHITE, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def mask_as_bgr(mask: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


def annotate_frame(
    bgr: np.ndarray,
    detections: list[GateDetection],
    poses: list[GatePose | None] | None = None,
    raceline: Raceline | None = None,
) -> np.ndarray:
    """Draw detections (green = selected next gate), corners, pose and raceline."""
    vis = bgr.copy()
    h, w = vis.shape[:2]

    if raceline is not None:
        cx, cy = raceline.centroid
        dx, dy = raceline.direction
        L = 120
        p1 = (int(cx - dx * L), int(cy - dy * L))
        p2 = (int(cx + dx * L), int(cy + dy * L))
        cv2.arrowedLine(vis, p2, p1, CYAN, 2, tipLength=0.2)

    for j, det in enumerate(detections):
        color = GREEN if j == 0 else AMBER
        cv2.polylines(vis, [det.corners.astype(np.int32)], True, color, 2, cv2.LINE_AA)
        for p in det.corners.astype(int):
            cv2.circle(vis, tuple(p), 4, MAGENTA, -1)
        if poses is not None and j < len(poses) and poses[j] is not None:
            pose = poses[j]
            cx, cy = det.center
            label = f"{pose.range_m:4.1f} m  {np.degrees(pose.bearing_rad):+.0f} deg"
            _put(vis, label, (int(cx) - 40, int(cy) - 10), 0.5, color)

    _put(vis, f"gates: {len(detections)}", (8, 22), 0.6, WHITE)
    return vis


def build_pipeline_panel(
    bgr: np.ndarray,
    gate_mask: np.ndarray,
    detections: list[GateDetection],
    poses: list[GatePose | None],
    raceline: Raceline | None = None,
) -> np.ndarray:
    """A 2x2 figure: (1) input, (2) HSV+morph mask, (3) contours/hierarchy,
    (4) inner-corner quad + PnP range."""
    h, w = bgr.shape[:2]

    # (1) input
    p1 = bgr.copy()
    _put(p1, "1. input frame (640x360)", (8, 22), 0.55)

    # (2) mask
    p2 = mask_as_bgr(gate_mask)
    _put(p2, "2. HSV gate mask + morphology", (8, 22), 0.55)

    # (3) contours and holes on top of the mask
    p3 = mask_as_bgr(gate_mask, (60, 60, 60))
    contours, hierarchy = cv2.findContours(gate_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            if hierarchy[i][3] == -1:
                cv2.drawContours(p3, [cnt], -1, AMBER, 1)       # outer gate frame
            else:
                cv2.drawContours(p3, [cnt], -1, GREEN, 1)       # inner hole
    _put(p3, "3. contours: frame (amber) / hole (green)", (8, 22), 0.5)

    # (4) selected gate corners + pose
    p4 = annotate_frame(bgr, detections, poses, raceline)
    _put(p4, "4. inner corners + PnP range", (8, h - 14), 0.55)

    top = np.hstack([p1, p2])
    bot = np.hstack([p3, p4])
    return np.vstack([top, bot])


def contact_sheet(tiles: list[np.ndarray], cols: int = 3, scale: float = 0.5) -> np.ndarray:
    """Tile a list of same-shape BGR images into a grid."""
    if not tiles:
        return np.zeros((10, 10, 3), dtype=np.uint8)
    small = [cv2.resize(t, (int(t.shape[1] * scale), int(t.shape[0] * scale))) for t in tiles]
    th, tw = small[0].shape[:2]
    rows = []
    for r in range(0, len(small), cols):
        chunk = small[r:r + cols]
        while len(chunk) < cols:
            chunk.append(np.zeros((th, tw, 3), dtype=np.uint8))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)
