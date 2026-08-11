"""Upload-mode pipeline: VideoFileSource -> ShotEngine -> EventLog (+ Renderer).

This is wiring only. Shot logic lives in the engine; drawing in the renderer;
file I/O here and in the event log. The upload *interface* (drag a clip, get
an mp4 + stats back) arrives at M5 and calls this function.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from engine.config import EngineConfig
from engine.engine import Detector, PoseModel, ShotEngine
from stats.event_log import EventLog
from stats.stats import session_summary


def run_upload(
    video_path: str | Path,
    detector: Detector,
    pose_model: Optional[PoseModel] = None,
    config: Optional[EngineConfig] = None,
    session_id: Optional[str] = None,
    sessions_root: str | Path = "sessions",
    renderer=None,
) -> dict:
    """Process one clip offline; returns the session summary dict.

    ``renderer`` stays None until M4; when set, annotated output writing is
    added here (never inside the engine).
    """
    from sources.video_file import VideoFileSource  # deferred: needs OpenCV

    config = config or EngineConfig.upload()
    session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
    session_dir = Path(sessions_root) / session_id

    engine = ShotEngine(config, detector=detector, pose_model=pose_model)

    with VideoFileSource(video_path) as source, EventLog(session_dir) as log:
        for frame, t in source.frames():
            state = engine.process_frame(frame, t)
            log.consume(state)
            if renderer is not None:
                renderer.draw(frame, state)
        # A shot still open when the clip ends is a real trajectory end.
        log.consume_events(engine.finalize())
        log.write_session_metadata(
            {
                "session_id": session_id,
                "source": str(video_path),
                "fps": source.fps,
                "mode": config.mode,
                "calibration": engine.calibration.to_metadata(),
            }
        )

    return session_summary(session_dir / "shots.jsonl")
