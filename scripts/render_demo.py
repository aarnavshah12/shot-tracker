"""M4 demo renderer CLI: full split-screen annotated video for clips.

Usage: .venv/bin/python scripts/render_demo.py [--out DIR] IMG_7101 [...]
       .venv/bin/python scripts/render_demo.py --out sessions/annotated --all
Writes <out>/<clip>_demo.mp4 (h264/yuv420p, source audio muxed); skips clips
already rendered so an interrupted batch resumes.
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
from render.renderer import Renderer, VideoSink

FOOTAGE = Path("footage/session-2026-08-11")


def render(stem: str, detector, pose_model, out_dir: Path) -> Path:
    cfg = EngineConfig.upload()
    src = FOOTAGE / f"{stem}.MOV"
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    engine = ShotEngine(cfg, detector=detector, pose_model=pose_model)
    renderer = Renderer(cfg, initials="AA", total_frames=total)
    sink = VideoSink(out_dir / f"{stem}_demo.mp4", fps)

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        state = engine.process_frame(frame, i / fps)
        sink.write(renderer.draw(frame, state))
        i += 1
    cap.release()
    engine.finalize()
    return sink.close(audio_source=src)


def main() -> None:
    args = sys.argv[1:]
    out_dir = Path("sessions/demo")
    if "--out" in args:
        i = args.index("--out")
        out_dir = Path(args[i + 1])
        del args[i : i + 2]
    if "--all" in args:
        stems = sorted(p.stem for p in FOOTAGE.glob("*.MOV"))
    else:
        stems = args or ["IMG_7101"]

    cfg = EngineConfig.upload()
    detector = RFDETRDetector(
        DETECTOR_V0_MODEL_ID, cfg.detector.ball_confidence, cfg.detector.rim_confidence
    )
    pose_model = RFDETRPoseModel()
    for stem in stems:
        target = out_dir / f"{stem}_demo.mp4"
        if target.exists():
            print(f"{target} (cached)", flush=True)
            continue
        print(render(stem, detector, pose_model, out_dir), flush=True)


if __name__ == "__main__":
    main()
