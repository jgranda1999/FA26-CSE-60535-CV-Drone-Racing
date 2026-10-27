"""
dronevision
===========

Part 3 "first solution" for the monocular gate-perception pipeline
(CSE 60535, Fall 2026 semester project).

This package implements the classical, no-training-required portion of the
Part 1 architecture:

    preprocessing  -> HSV color segmentation + morphology
    detector       -> contour / hierarchy shape filter -> inner-gate corners
    raceline       -> cyan look-ahead direction
    pose           -> PnP on the 1.5 m inner square -> gate pose in camera frame
    visualize      -> overlays and report figures

Camera intrinsics and gate geometry come from the simulator spec
(VADR-TS-003, issue 00.03):

    image           640 x 360, pinhole, no distortion
    (cx, cy)        (320, 180) px
    (fx, fy)        (320, 320) px
    gate inner sq.  1500 mm x 1500 mm   (the PnP target)
    gate outer sq.  2700 mm x 2700 mm
    camera tilt     20 deg upward from the body x-axis
"""

from .preprocessing import CameraModel, gate_color_mask, raceline_color_mask, load_run
from .detector import GateDetection, detect_gates
from .raceline import estimate_raceline
from .pose import estimate_gate_pose

__all__ = [
    "CameraModel",
    "gate_color_mask",
    "raceline_color_mask",
    "load_run",
    "GateDetection",
    "detect_gates",
    "estimate_raceline",
    "estimate_gate_pose",
]
