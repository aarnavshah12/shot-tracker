"""Pose-era debug renderer: runs detector + pose + engine live on a clip and
draws skeleton, ball trail, rim, phase, verdict, and release form metrics.

Still NOT the M4 renderer — a richer debug view while pose lands.

Usage: .venv/bin/python scripts/render_pose_debug.py IMG_7101 [IMG_7097 ...]
Writes sessions/previews/<clip>_pose_debug.mp4.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from engine.config import EngineConfig
from engine.engine import ShotEngine
from models.detector import RFDETRDetector
from models.pose import RFDETRPoseModel
from models.registry import DETECTOR_V0_MODEL_ID

FOOTAGE = Path("footage/session-2026-08-11")
OUT = Path("sessions/previews")

GREEN = (80, 200, 80)
BLUE = (255, 120, 40)
ORANGE = (0, 140, 255)
RED = (60, 60, 230)
WHITE = (245, 245, 245)
PINK = (180, 105, 255)

SKELETON = (
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)


def draw_pose(frame, pose, min_conf: float) -> None:
    kp = pose.keypoints
    for a, b in SKELETON:
        pa, pb = kp.get(a), kp.get(b)
        if pa and pb and pa[2] >= min_conf and pb[2] >= min_conf:
            cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), PINK, 3)
    for name, (x, y, c) in kp.items():
        if c >= min_conf and not name.startswith(("nose", "left_e", "right_e")):
            cv2.circle(frame, (int(x), int(y)), 5, PINK, -1)


def fmt(v, unit=""):
    return "null" if v is None else f"{v:.1f}{unit}"


def render(stem: str) -> Path:
    cfg = EngineConfig.upload()
    detector = RFDETRDetector(
        DETECTOR_V0_MODEL_ID, cfg.detector.ball_confidence, cfg.detector.rim_confidence
    )
    pose_model = RFDETRPoseModel()
    engine = ShotEngine(cfg, detector=detector, pose_model=pose_model)

    cap = cv2.VideoCapture(str(FOOTAGE / f"{stem}.MOV"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{stem}_pose_debug.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    banner = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        st = engine.process_frame(frame, i / fps)

        if st.rim:
            x1, y1, x2, y2 = (int(v) for v in st.rim.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            cv2.putText(frame, "rim", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREEN, 2)
        if st.pose:
            draw_pose(frame, st.pose, cfg.pose.keypoint_confidence)
        for a, b in zip(st.trail, st.trail[1:]):
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), ORANGE, 3)
        if st.ball:
            color = (160, 160, 160) if st.ball.interpolated else BLUE
            cv2.circle(frame, (int(st.ball.x), int(st.ball.y)), 14, color, 3)

        cv2.putText(frame, st.phase.value.upper(), (24, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, WHITE, 3)
        if st.current_speed_ms is not None:
            cv2.putText(frame, f"{st.current_speed_ms * 3.6:4.0f} KM/H", (24, h - 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
        cv2.putText(frame, f"FRAME {i} / {n - 1}", (w - 320, h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, WHITE, 2)

        for e in st.events:
            text = "MAKE!!" if e.verdict.value == "make" else "MISS"
            color = GREEN if e.verdict.value == "make" else RED
            lines = [
                f"{text} ({e.verdict_confidence.value})",
                f"elbow {fmt(e.elbow_deg, ' deg')}   knee {fmt(e.knee_deg, ' deg')}",
                f"tilt {fmt(e.shoulder_hip_deg, ' deg')}   rel height {fmt(e.release_height_m, ' m')}",
                f"rel speed {fmt(e.release_velocity_ms, ' m/s')}   entry {fmt(e.entry_angle_deg, ' deg')}",
            ]
            banner = (lines, color, int(fps * 2.5))
        if banner and banner[2] > 0:
            lines, color, ttl = banner
            cv2.putText(frame, lines[0], (24, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 6)
            for k, line in enumerate(lines[1:], start=1):
                cv2.putText(frame, line, (24, 120 + 46 * k),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.95, WHITE, 2)
            banner = (lines, color, ttl - 1)

        writer.write(frame)
        i += 1

    cap.release()
    for e in engine.finalize():
        pass  # end-of-stream shots would be reflected in logs, not this video
    writer.release()
    return out_path


if __name__ == "__main__":
    for stem in sys.argv[1:] or ["IMG_7101"]:
        print(render(stem), flush=True)
