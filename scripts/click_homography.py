"""Owner tool: click 4 known ground points to calibrate the shot chart.

Setup (plan section 5): put 4 chalk marks on the court at known distances
from the point directly under the rim center (e.g. free-throw line center,
two elbows, a corner mark). Measure their court X/Y in meters:
X = sideways (right of the hoop positive), Y = distance out from the hoop.

Usage:
  .venv/bin/python scripts/click_homography.py footage/session-2026-08-11/IMG_7101.MOV \
      --points "0,4.6" "-1.8,4.6" "1.8,4.6" "0,6.75"

A window opens on the clip's first frame; click the 4 marks IN THE SAME ORDER
as the --points list. Saves sessions/homography.json (used by the engine for
shooter court position; without it the shot chart stays disabled and
everything else runs).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument(
        "--points", nargs=4, required=True,
        help='4 court points as "x_m,y_m" in click order',
    )
    ap.add_argument("--out", default="sessions/homography.json")
    args = ap.parse_args()

    court = [tuple(float(v) for v in p.split(",")) for p in args.points]
    cap = cv2.VideoCapture(args.video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not read a frame from {args.video}")

    clicks: list[tuple[int, int]] = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append((x, y))

    win = "click 4 ground marks in order (q to abort)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        for i, (x, y) in enumerate(clicks):
            cv2.circle(vis, (x, y), 8, (0, 140, 255), -1)
            cv2.putText(vis, f"{i + 1}: {court[i]}", (x + 12, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            raise SystemExit("aborted")
        if len(clicks) == 4:
            break
    cv2.destroyAllWindows()

    H, _ = cv2.findHomography(
        np.array(clicks, dtype=np.float64), np.array(court, dtype=np.float64)
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {
            "video": str(args.video),
            "image_points": clicks,
            "court_points_m": court,
            "homography": H.tolist(),
        },
        indent=2,
    ))
    print(f"saved {out}")
    print("verification (each click -> court meters):")
    for (x, y), c in zip(clicks, court):
        v = H @ np.array([x, y, 1.0])
        print(f"  ({x},{y}) -> ({v[0] / v[2]:+.2f}, {v[1] / v[2]:+.2f})  expected {c}")


if __name__ == "__main__":
    main()
