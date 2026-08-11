"""Quick debug-overlay renderer: draws cached diagnostics onto clips.

NOT the M4 renderer (that's the AA-style split-screen with pose, panels, and
styling, built in render/). This just replays sessions/m1-diag JSON onto the
video so results are visible: rim box, ball + trail, phase, verdict banner.

Usage: .venv/bin/python scripts/render_debug.py IMG_7101 [IMG_7090 ...]
Writes sessions/previews/<clip>_debug.mp4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

FOOTAGE = Path("footage/session-2026-08-11")
DIAG = Path("sessions/m1-diag")
OUT = Path("sessions/previews")

GREEN = (80, 200, 80)
BLUE = (255, 120, 40)
ORANGE = (0, 140, 255)
RED = (60, 60, 230)
WHITE = (245, 245, 245)


def render(stem: str) -> Path:
    d = json.loads((DIAG / f"{stem}.json").read_text())
    rows, events = d["rows"], d["events"]
    verdict_at = {e["frames"][1]: e for e in events}

    cap = cv2.VideoCapture(str(FOOTAGE / f"{stem}.MOV"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{stem}_debug.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    banner = None  # (text, color, frames_left)
    trail: list[tuple[int, int]] = []
    n = len(rows)
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        r = rows[i]

        if r["rim"]:
            x1, y1, x2, y2 = (int(v) for v in r["rim"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(frame, "rim", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)

        if r["ball"]:
            bx, by = int(r["ball"][0]), int(r["ball"][1])
            trail.append((bx, by))
            trail = trail[-20:]
            for a, b in zip(trail, trail[1:]):
                cv2.line(frame, a, b, ORANGE, 3)
            color = (160, 160, 160) if r["ball_interp"] else BLUE
            cv2.circle(frame, (bx, by), 14, color, 3)
        else:
            trail = []

        cv2.putText(frame, r["phase"].upper(), (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
        cv2.putText(frame, f"FRAME {i} / {n - 1}", (w - 320, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

        if i in verdict_at:
            e = verdict_at[i]
            text = "MAKE!!" if e["verdict"] == "make" else "MISS"
            color = GREEN if e["verdict"] == "make" else RED
            banner = (f"{text} ({e['verdict_confidence']})", color, int(fps * 2))
        if banner and banner[2] > 0:
            cv2.putText(frame, banner[0], (24, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, banner[1], 6)
            banner = (banner[0], banner[1], banner[2] - 1)

        writer.write(frame)

    cap.release()
    writer.release()
    return out_path


if __name__ == "__main__":
    for stem in sys.argv[1:] or ["IMG_7101"]:
        print(render(stem))
