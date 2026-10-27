#!/usr/bin/env python3
"""
run_part3.py - run the Part 3 classical gate-perception pipeline on one logged
simulator run and produce the report figures + a per-frame results CSV.

Example
-------
    python run_part3.py --input ../sample-data/run_20260731_004531.zip \
                        --outdir out

Outputs (under --outdir):
    results.csv              per-frame detections and PnP estimates
    fig_pipeline.jpg         stage-by-stage panel on one representative frame
    fig_detections.jpg       detection gallery across the run
    fig_range_vs_time.png    PnP range to the selected gate vs time, by attempt
    annotated/ (optional)    every processed frame with overlays (--save-frames)

The pipeline is classical and training-free: HSV color segmentation ->
morphology -> contour/hierarchy shape filter -> inner-square corners -> PnP.
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from dronevision import (
    CameraModel,
    detect_gates,
    estimate_gate_pose,
    estimate_raceline,
    load_run,
)
from dronevision.visualize import annotate_frame, build_pipeline_panel, contact_sheet


def assign_attempts(gate_index: pd.Series) -> np.ndarray:
    """Segment attempts: a drop in active gate index (e.g. 1 -> 0) is a restart."""
    g = gate_index.to_numpy()
    attempt = np.zeros(len(g), dtype=int)
    cur = 0
    for i in range(1, len(g)):
        if g[i] < g[i - 1]:
            cur += 1
        attempt[i] = cur
    return attempt


def process(args) -> None:
    os.makedirs(args.outdir, exist_ok=True)
    camera = CameraModel()
    run = load_run(args.input)
    frames = run.frames.copy()
    frames["attempt"] = assign_attempts(frames["gate_index"])
    n = len(frames)
    print(f"Loaded run with {n} frames from {run.root}")

    if args.save_frames:
        os.makedirs(os.path.join(args.outdir, "annotated"), exist_ok=True)

    records = []
    gallery_tiles = []
    gallery_idxs = set(np.linspace(0, n - 1, args.gallery, dtype=int).tolist())
    pipeline_panel = None
    best_panel_score = -1.0

    for i, row in frames.iterrows():
        bgr = run.read_frame(row)
        detections, gate_mask = detect_gates(bgr)
        raceline, _ = estimate_raceline(bgr)

        poses = [estimate_gate_pose(d.corners, camera) for d in detections]
        sel = detections[0] if detections else None
        sel_pose = poses[0] if poses else None

        records.append(dict(
            frame_id=int(row["frame_id"]),
            t_rel=float(row["t_rel"]),
            attempt=int(row["attempt"]),
            active_gate=int(row["gate_index"]),
            n_gates=len(detections),
            sel_inner_area=(sel.inner_area if sel else np.nan),
            sel_touches_border=(sel.touches_border if sel else False),
            range_m=(sel_pose.range_m if sel_pose else np.nan),
            bearing_deg=(np.degrees(sel_pose.bearing_rad) if sel_pose else np.nan),
            elevation_deg=(np.degrees(sel_pose.elevation_rad) if sel_pose else np.nan),
            reproj_err_px=(sel_pose.reproj_error_px if sel_pose else np.nan),
            raceline_offset_px=(raceline.bottom_offset_px if raceline else np.nan),
        ))

        if i in gallery_idxs:
            vis = annotate_frame(bgr, detections, poses, raceline)
            from dronevision.visualize import _put
            _put(vis, f"t={row['t_rel']:.1f}s g{int(row['gate_index'])}", (8, 348), 0.5)
            gallery_tiles.append(vis)

        # Choose a representative frame for the stage panel: a clean mid-range
        # single-gate detection with low reprojection error.
        if sel is not None and sel_pose is not None and not sel.touches_border:
            panel_score = sel.rectangularity - 0.05 * sel_pose.reproj_error_px
            if 1500 < sel.inner_area < 6000 and panel_score > best_panel_score:
                best_panel_score = panel_score
                pipeline_panel = build_pipeline_panel(bgr, gate_mask, detections, poses, raceline)

        if args.save_frames:
            vis = annotate_frame(bgr, detections, poses, raceline)
            cv2.imwrite(os.path.join(args.outdir, "annotated", row["file"]), vis)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{n}")

    df = pd.DataFrame.from_records(records)
    csv_path = os.path.join(args.outdir, "results.csv")
    df.to_csv(csv_path, index=False)

    # ----------------------------- figures ------------------------------- #
    if pipeline_panel is not None:
        cv2.imwrite(os.path.join(args.outdir, "fig_pipeline.jpg"), pipeline_panel)
    if gallery_tiles:
        sheet = contact_sheet(gallery_tiles, cols=3, scale=0.6)
        cv2.imwrite(os.path.join(args.outdir, "fig_detections.jpg"), sheet)
    _range_vs_time_plot(df, run, os.path.join(args.outdir, "fig_range_vs_time.png"))

    _print_summary(df)
    print(f"\nWrote results and figures to {os.path.abspath(args.outdir)}/")


def _range_vs_time_plot(df: pd.DataFrame, run, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"(skipping range-vs-time plot: matplotlib unavailable: {exc})")
        return

    valid = df[df["range_m"].notna()]
    fig, ax = plt.subplots(figsize=(8.5, 4.3))

    # Scatter the PnP range, colored by which gate is active. No connecting
    # lines, so the gate pass (a discontinuity) is not drawn as a vertical jump.
    g0 = valid[valid["active_gate"] == 0]
    g1 = valid[valid["active_gate"] == 1]
    ax.scatter(g0["t_rel"], g0["range_m"], s=14, c="#1f77b4",
               label="gate 0 active (approach)")
    ax.scatter(g1["t_rel"], g1["range_m"], s=14, c="#d62728",
               label="gate 1 active (re-acquired, farther)")

    # Mark the start of each new attempt (active gate index resets).
    first = True
    for att, sub in df.groupby("attempt"):
        if att == 0:
            continue
        t0 = sub["t_rel"].min()
        ax.axvline(t0, color="gray", ls="--", lw=1,
                   label="attempt restart" if first else None)
        first = False

    ax.set_xlabel("time since first frame (s)")
    ax.set_ylabel("PnP range to selected gate (m)")
    ax.set_title("Per-attempt: range to gate 0 falls on approach, then gate 1 is\n"
                 "re-acquired at a larger range")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def _print_summary(df: pd.DataFrame) -> None:
    n = len(df)
    det = int((df["n_gates"] > 0).sum())
    active0 = df[df["active_gate"] == 0]
    det0 = int((active0["n_gates"] > 0).sum())
    print("\n================ Part 3 pipeline summary ================")
    print(f"frames processed ............. {n}")
    print(f"frames with >=1 gate ......... {det} ({100*det/max(n,1):.1f}%)")
    print(f"frames while gate 0 active ... {len(active0)}")
    print(f"  of which detected .......... {det0} ({100*det0/max(len(active0),1):.1f}%)")
    valid = df["range_m"].notna()
    if valid.any():
        print(f"median PnP reproj error ...... {df.loc[valid,'reproj_err_px'].median():.2f} px")
        print(f"PnP range (selected) ......... "
              f"{df.loc[valid,'range_m'].min():.1f} - {df.loc[valid,'range_m'].max():.1f} m")
    print(f"mean gates per frame ......... {df['n_gates'].mean():.2f}")
    print("=========================================================")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="run .zip or run directory")
    ap.add_argument("--outdir", default="out", help="output directory")
    ap.add_argument("--gallery", type=int, default=12,
                    help="number of frames in the detection gallery figure")
    ap.add_argument("--save-frames", action="store_true",
                    help="also write every annotated frame to outdir/annotated/")
    process(ap.parse_args())


if __name__ == "__main__":
    main()
