"""M1 diagnostics: run detector v0 + engine over every clip in a session
folder, dumping per-frame rows and events for acceptance analysis.

Acceptance (plan M1): ball detected in >= 90% of in-flight frames; rim box
stable (IoU with the session median box >= 0.8 across the session).

Usage: .venv/bin/python scripts/m1_diagnostics.py [footage_dir] [out_dir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from engine.config import EngineConfig
from engine.engine import ShotEngine
from models.detector import RFDETRDetector
from models.pose import RFDETRPoseModel
from models.registry import DETECTOR_V0_MODEL_ID


def process_clip(clip: Path, detector, cfg: EngineConfig, pose_model=None) -> dict:
    engine = ShotEngine(cfg, detector=detector, pose_model=pose_model)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    rows, events = [], []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        st = engine.process_frame(frame, i / fps)
        rows.append(
            {
                "f": i,
                "pose": st.pose is not None and not st.pose_stale,
                "ball": [round(st.ball.x, 1), round(st.ball.y, 1)] if st.ball else None,
                "ball_interp": st.ball.interpolated if st.ball else None,
                "ball_vy": round(st.ball.vy, 1) if st.ball else None,
                "rim": [round(v, 1) for v in st.rim.bbox] if st.rim else None,
                "rim_conf": round(st.rim.confidence, 3) if st.rim else None,
                "phase": st.phase.value,
                "scale": round(st.scale_px_per_m, 1) if st.scale_px_per_m else None,
            }
        )
        events.extend(e.to_json_dict() for e in st.events)
        i += 1
    cap.release()
    flushed = [e.to_json_dict() for e in engine.finalize()]
    events.extend(flushed)
    return {
        "clip": clip.name,
        "fps": fps,
        "frames": i,
        "rows": rows,
        "events": events,
        "flushed_events": len(flushed),
        "calibration": engine.calibration.to_metadata(),
    }


def main() -> None:
    footage = Path(sys.argv[1] if len(sys.argv) > 1 else "footage/session-2026-08-11")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "sessions/m1-diag")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = EngineConfig.upload()
    detector = RFDETRDetector(
        DETECTOR_V0_MODEL_ID,
        ball_confidence=cfg.detector.ball_confidence,
        rim_confidence=cfg.detector.rim_confidence,
    )
    pose_model = RFDETRPoseModel() if cfg.pose.enabled else None

    for clip in sorted(footage.glob("*.MOV")):
        out_file = out_dir / (clip.stem + ".json")
        if out_file.exists():
            print(f"{clip.name}: cached", flush=True)
            continue
        result = process_clip(clip, detector, cfg, pose_model)
        out_file.write_text(json.dumps(result))
        print(
            f"{clip.name}: {result['frames']} frames, {len(result['events'])} events",
            flush=True,
        )
    print("DONE")


if __name__ == "__main__":
    main()
