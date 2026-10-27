"""
Stage 1 - data preprocessing and color segmentation.

Two responsibilities:

1. `load_run` reads a logged simulator run (a directory or a .zip) and returns
   the per-frame table joined with the active-gate label, so the rest of the
   pipeline can iterate over frames in time order.
2. `gate_color_mask` / `raceline_color_mask` turn a BGR frame into the binary
   masks the detector works on, using HSV thresholding plus morphology.

The HSV thresholds were tuned on frames from `run_20260731_004531`. The gate is
a saturated orange-red, which straddles the hue wrap-around at 0/180 in
OpenCV's H in [0, 179] convention, so two hue bands are OR-ed together.
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Camera model (from the simulator spec, VADR-TS-003)
# --------------------------------------------------------------------------- #
@dataclass
class CameraModel:
    """Pinhole camera, no lens distortion (simulator spec 3.8)."""

    width: int = 640
    height: int = 360
    fx: float = 320.0
    fy: float = 320.0
    cx: float = 320.0
    cy: float = 180.0

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @property
    def dist(self) -> np.ndarray:
        # No distortion in the simulated pinhole camera.
        return np.zeros((5, 1), dtype=np.float64)


# --------------------------------------------------------------------------- #
# HSV segmentation thresholds
# --------------------------------------------------------------------------- #
@dataclass
class GateColorThresholds:
    """Orange-red gate frame. Red wraps around H = 0/180 so we use two bands."""

    lo1: tuple = (0, 90, 80)
    hi1: tuple = (14, 255, 255)
    lo2: tuple = (166, 90, 80)
    hi2: tuple = (179, 255, 255)
    close_ksize: int = 5
    close_iter: int = 3
    open_iter: int = 1


@dataclass
class RacelineColorThresholds:
    """Bright cyan raceline ribbon."""

    lo: tuple = (85, 70, 90)
    hi: tuple = (105, 255, 255)
    close_ksize: int = 5
    close_iter: int = 2


def gate_color_mask(bgr: np.ndarray, th: GateColorThresholds | None = None) -> np.ndarray:
    """Binary mask of the orange-red gate frames, after morphological cleanup.

    `MORPH_CLOSE` seals the thin gate border (which is interrupted by printed
    text, logos and the gate ID) so that the inner square shows up as an
    enclosed hole for the detector. `MORPH_OPEN` then removes small speckle
    such as red signage and start lights.
    """
    th = th or GateColorThresholds()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, th.lo1, th.hi1)
    m2 = cv2.inRange(hsv, th.lo2, th.hi2)
    mask = cv2.bitwise_or(m1, m2)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th.close_ksize, th.close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=th.close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=th.open_iter)
    return mask


def raceline_color_mask(bgr: np.ndarray, th: RacelineColorThresholds | None = None) -> np.ndarray:
    """Binary mask of the cyan raceline ribbon."""
    th = th or RacelineColorThresholds()
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, th.lo, th.hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th.close_ksize, th.close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=th.close_iter)
    return mask


# --------------------------------------------------------------------------- #
# Run loading
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    """A loaded simulator run ready for frame-by-frame replay."""

    root: str                     # directory holding frames/ and the CSVs
    frames: pd.DataFrame          # frame_id, sim_time_ns, rx_wall, file, gate_index, t_rel
    race_status: pd.DataFrame
    _tmp: object = field(default=None, repr=False)  # keeps a TemporaryDirectory alive

    def frame_path(self, row) -> str:
        return os.path.join(self.root, "frames", row["file"])

    def read_frame(self, row) -> np.ndarray:
        img = cv2.imread(self.frame_path(row), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.frame_path(row))
        return img

    def __len__(self) -> int:
        return len(self.frames)


def _find_run_root(base: str) -> str:
    """Locate the directory that actually contains frames/ and frames.csv."""
    if os.path.isdir(os.path.join(base, "frames")):
        return base
    for name in sorted(os.listdir(base)):
        cand = os.path.join(base, name)
        if os.path.isdir(cand) and os.path.isdir(os.path.join(cand, "frames")):
            return cand
    raise FileNotFoundError(f"Could not find a 'frames/' folder under {base}")


def load_run(path: str) -> Run:
    """Load a run from either a directory or a .zip archive.

    The returned `Run.frames` table is sorted by `sim_time_ns` and carries a
    `t_rel` column (seconds since the first frame) for plotting.
    """
    tmp = None
    if path.lower().endswith(".zip"):
        tmp = tempfile.TemporaryDirectory(prefix="dronevision_")
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp.name)
        root = _find_run_root(tmp.name)
    else:
        root = _find_run_root(path)

    frames = pd.read_csv(os.path.join(root, "frames.csv"))
    frames = frames.sort_values("sim_time_ns").reset_index(drop=True)
    frames["t_rel"] = (frames["sim_time_ns"] - frames["sim_time_ns"].min()) / 1e9

    race_status_path = os.path.join(root, "race_status.csv")
    race_status = pd.read_csv(race_status_path) if os.path.exists(race_status_path) else pd.DataFrame()

    return Run(root=root, frames=frames, race_status=race_status, _tmp=tmp)
