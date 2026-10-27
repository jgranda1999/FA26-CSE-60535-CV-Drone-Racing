"""
Stage 3 - raceline look-ahead direction.

The cyan ribbon on the floor points along the course. When the next gate is
small or off-screen, its heading is still a useful look-ahead cue (Part 1,
"the raceline gives a look-ahead direction").

We mask the cyan ribbon, keep its largest connected component in the lower part
of the image (to ignore ceiling reflections), and fit a single line with
`cv2.fitLine`. The result is the ribbon centroid and a unit direction, plus the
signed horizontal offset of the ribbon at the bottom of the image (a crude
"steer left / right" cue).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import RacelineColorThresholds, raceline_color_mask


@dataclass
class Raceline:
    centroid: tuple          # (x, y) px
    direction: tuple         # unit (dx, dy), pointing "up" the image (toward the horizon)
    bottom_offset_px: float  # ribbon x at image bottom minus image center x
    area: float              # cyan pixel count of the kept component


def estimate_raceline(
    bgr: np.ndarray,
    thresholds: RacelineColorThresholds | None = None,
    min_area: float = 400.0,
    mask: np.ndarray | None = None,
) -> tuple[Raceline | None, np.ndarray]:
    """Estimate the raceline direction in a single frame. May return None."""
    h, w = bgr.shape[:2]
    if mask is None:
        mask = raceline_color_mask(bgr, thresholds)

    # Only trust the lower 65% of the image (floor ribbon, not reflections).
    roi = mask.copy()
    roi[: int(0.35 * h), :] = 0

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(roi, 8)
    if n <= 1:
        return None, mask
    # Largest non-background component.
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = float(stats[idx, cv2.CC_STAT_AREA])
    if area < min_area:
        return None, mask

    ys, xs = np.where(labels == idx)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    # Make the direction point toward the horizon (negative y / up the image).
    if vy > 0:
        vx, vy = -vx, -vy

    # Where does the fitted line cross the bottom row? (x at y = h - 1)
    if abs(vy) > 1e-6:
        t = (h - 1 - y0) / vy
        x_bottom = x0 + t * vx
    else:
        x_bottom = x0
    bottom_offset = float(x_bottom - w / 2.0)

    return (
        Raceline(
            centroid=(float(x0), float(y0)),
            direction=(float(vx), float(vy)),
            bottom_offset_px=bottom_offset,
            area=area,
        ),
        mask,
    )
