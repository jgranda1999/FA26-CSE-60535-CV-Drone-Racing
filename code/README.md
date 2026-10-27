# Part 3 code - classical gate perception pipeline

Training-free first solution for the monocular gate detector:

```
preprocessing  ->  HSV color segmentation + morphology      (dronevision/preprocessing.py)
detector       ->  contour/hierarchy shape filter -> corners (dronevision/detector.py)
raceline       ->  cyan look-ahead direction                 (dronevision/raceline.py)
pose           ->  PnP on the 1.5 m inner square             (dronevision/pose.py)
visualize      ->  overlays + report figures                 (dronevision/visualize.py)
run_part3.py   ->  command-line entry point
```

## Install

```bash
cd code
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Needs Python 3.10+.

## Run on the sample file

A sample run is included in this repo at `../sample-data/run_20260731_004531.zip`
(the same run analyzed in Appendix A). `run_part3.py` accepts either the `.zip`
directly or an already-extracted run directory.

```bash
cd code
python run_part3.py --input ../sample-data/run_20260731_004531.zip --outdir out
```

Add `--save-frames` to also dump every annotated frame to `out/annotated/`.

## Outputs (written to `--outdir`, default `out/`)

| File                    | Contents                                                        |
| ----------------------- | --------------------------------------------------------------- |
| `results.csv`           | per-frame detections + PnP range/bearing/elevation/reproj error |
| `fig_pipeline.jpg`      | stage-by-stage panel (input -> mask -> contours -> corners+PnP) |
| `fig_detections.jpg`    | detection gallery sampled across the run                        |
| `fig_range_vs_time.png` | PnP range to the selected gate vs time, per attempt             |

The script also prints a summary (detection rate, median reprojection error,
PnP range span). No model weights are used or produced.

## Use as a library

```python
from dronevision import load_run, detect_gates, estimate_gate_pose, CameraModel

run = load_run("../sample-data/run_20260731_004531.zip")
cam = CameraModel()
row = run.frames.iloc[100]
dets, mask = detect_gates(run.read_frame(row))
if dets:
    pose = estimate_gate_pose(dets[0].corners, cam)
    print(pose.range_m, pose.bearing_rad)
```
