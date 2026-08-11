"""Summarize M1 diagnostics into the acceptance metrics.

Reads the per-clip JSON dumps from scripts/m1_diagnostics.py and reports:
- ball detection rate inside flight windows (plan: >= 90% of in-flight frames)
- rim stability: IoU of each raw rim detection vs the session median box
  (plan: >= 0.8 across the session)
- events per clip, plus calibration end-state for debugging.

Usage: .venv/bin/python scripts/m1_summarize.py [diag_dir]
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main() -> None:
    diag_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "sessions/m1-diag")
    all_flight_frames = all_flight_detected = 0
    all_rim_ious: list[float] = []

    for f in sorted(diag_dir.glob("*.json")):
        d = json.loads(f.read_text())
        rows, events = d["rows"], d["events"]

        rim_boxes = [r["rim"] for r in rows if r["rim"]]
        rim_ious = []
        if len(rim_boxes) >= 3:
            med = [statistics.median(b[i] for b in rim_boxes) for i in range(4)]
            rim_ious = [iou(b, med) for b in rim_boxes]
            all_rim_ious.extend(rim_ious)

        flight_frames = flight_detected = 0
        for e in events:
            f0, f1 = e["frames"]
            for r in rows[f0 : f1 + 1]:
                flight_frames += 1
                if r["ball"] and not r["ball_interp"]:
                    flight_detected += 1
        all_flight_frames += flight_frames
        all_flight_detected += flight_detected

        rim_cov = len(rim_boxes) / max(1, d["frames"])
        rim_stable = (
            sum(1 for v in rim_ious if v >= 0.8) / len(rim_ious) if rim_ious else 0.0
        )
        ev_str = (
            "; ".join(
                f"{e['verdict']}/{e['verdict_confidence']} f{e['frames'][0]}-{e['frames'][1]}"
                f" v={e['release_velocity_ms']}m/s ang={e['entry_angle_deg']}"
                for e in events
            )
            or "none"
        )
        flight_str = (
            f"{flight_detected}/{flight_frames} ({100 * flight_detected / flight_frames:.0f}%)"
            if flight_frames
            else "n/a"
        )
        cal = d["calibration"]
        print(f"{d['clip']}: {d['frames']}f | ball-in-flight {flight_str} | "
              f"rim cov {rim_cov:.0%} IoU>=0.8 {rim_stable:.0%} "
              f"min-IoU {min(rim_ious):.2f} | cal samples={cal['samples']} "
              f"px/m={cal['px_per_m'] and round(cal['px_per_m'])} | {ev_str}")

    print("\n=== AGGREGATE (M1 acceptance) ===")
    if all_flight_frames:
        pct = 100 * all_flight_detected / all_flight_frames
        print(f"ball detected in flight: {all_flight_detected}/{all_flight_frames} "
              f"({pct:.1f}%)  [target >= 90%]")
    if all_rim_ious:
        ok = sum(1 for v in all_rim_ious if v >= 0.8) / len(all_rim_ious)
        print(f"rim IoU vs session median >= 0.8: {ok:.1%} of detections "
              f"(median IoU {statistics.median(all_rim_ious):.3f})  [target: stable]")


if __name__ == "__main__":
    main()
