"""
Stage 2 - classical gate detector.

Given the gate color mask, find the race gates and, for each, the four corners
of its *inner* square (the clean PnP target identified in Part 1, Appendix A).

Method
------
1. `findContours` with `RETR_CCOMP` so each red blob is returned as an outer
   contour (the gate frame) together with its holes (level-1 children).
2. For every outer contour large enough to matter, take its largest hole:
   the enclosed region bounded by the gate frame, i.e. the inner square.
3. Fit a rotated rectangle (`minAreaRect`) to that hole. The rectangle gives
   four ordered corners even when printed text / checker patterns nibble the
   inner edge, which a strict 4-point `approxPolyDP` would reject.
4. Shape-filter the candidate on aspect ratio and rectangularity, and score it
   so the controller can pick the *next* gate (biggest, most rectangular,
   most square).

No training and no labels are required; everything is derived from the known
color and shape of the gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import GateColorThresholds, gate_color_mask


@dataclass
class GateDetection:
    corners: np.ndarray        # (4, 2) float32, ordered TL, TR, BR, BL
    center: tuple              # (cx, cy) of the inner square, px
    inner_area: float          # hole area, px^2
    frame_area: float          # red-ring area, px^2
    aspect_ratio: float        # >= 1
    rectangularity: float      # hole_area / minAreaRect_area in [0, 1]
    score: float               # higher = more gate-like / nearer
    touches_border: bool       # True if the quad hits the image edge


# --------------------------------------------------------------------------- #
# Corner ordering
# --------------------------------------------------------------------------- #
def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL using the sum / difference trick."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()      # x - y ... actually y - x per column order
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _touches_border(corners: np.ndarray, w: int, h: int, margin: int = 2) -> bool:
    x = corners[:, 0]
    y = corners[:, 1]
    return bool(
        (x <= margin).any() or (x >= w - 1 - margin).any()
        or (y <= margin).any() or (y >= h - 1 - margin).any()
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_gates(
    bgr: np.ndarray,
    thresholds: GateColorThresholds | None = None,
    min_frame_area: float = 250.0,
    min_inner_area: float = 100.0,
    max_aspect_ratio: float = 1.8,
    min_rectangularity: float = 0.60,
    mask: np.ndarray | None = None,
) -> tuple[list[GateDetection], np.ndarray]:
    """Detect gates in a single BGR frame.

    Returns `(detections, mask)` where detections are sorted best-first; the
    first detection is the pipeline's guess at the *next* gate to fly through.
    """
    h, w = bgr.shape[:2]
    if mask is None:
        mask = gate_color_mask(bgr, thresholds)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[GateDetection] = []
    if hierarchy is None:
        return detections, mask
    hierarchy = hierarchy[0]

    for i, cnt in enumerate(contours):
        if hierarchy[i][3] != -1:          # keep only outer contours (no parent)
            continue
        frame_area = cv2.contourArea(cnt)
        if frame_area < min_frame_area:
            continue

        # Collect child holes of this outer contour.
        child = hierarchy[i][2]
        holes = []
        while child != -1:
            holes.append(child)
            child = hierarchy[child][0]
        if not holes:
            continue

        hole = max(holes, key=lambda j: cv2.contourArea(contours[j]))
        inner_area = cv2.contourArea(contours[hole])
        if inner_area < min_inner_area:
            continue

        (rcx, rcy), (rw, rh), _ang = cv2.minAreaRect(contours[hole])
        if rw < 1.0 or rh < 1.0:
            continue
        aspect_ratio = max(rw, rh) / min(rw, rh)
        rectangularity = inner_area / (rw * rh)
        if aspect_ratio > max_aspect_ratio or rectangularity < min_rectangularity:
            continue

        corners = order_quad(cv2.boxPoints(((rcx, rcy), (rw, rh), _ang)))
        # Score: larger (nearer) and more square / rectangular rank higher.
        score = inner_area * rectangularity / aspect_ratio
        detections.append(
            GateDetection(
                corners=corners,
                center=(float(rcx), float(rcy)),
                inner_area=float(inner_area),
                frame_area=float(frame_area),
                aspect_ratio=float(aspect_ratio),
                rectangularity=float(rectangularity),
                score=float(score),
                touches_border=_touches_border(corners, w, h),
            )
        )

    detections.sort(key=lambda d: d.score, reverse=True)
    return detections, mask
