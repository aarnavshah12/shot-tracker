"""Upload-mode pipeline: VideoFileSource -> ShotEngine -> EventLog + Renderer.

This is wiring only. Shot logic lives in the engine; drawing in the renderer;
file I/O here and in the event log/sink. The upload interface (app/server.py)
calls run_upload with a progress callback.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

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
    render_video: bool = True,
    initials: str = "AA",
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Process one clip offline; returns paths plus the session summary."""
    from sources.video_file import VideoFileSource  # deferred: needs OpenCV

    config = config or EngineConfig.upload()
    session_id = session_id or time.strftime("%Y%m%d-%H%M%S")
    session_dir = Path(sessions_root) / session_id

    engine = ShotEngine(config, detector=detector, pose_model=pose_model)
    annotated: Optional[Path] = None

    with VideoFileSource(video_path) as source, EventLog(session_dir) as log:
        renderer = sink = None
        if render_video:
            from render.renderer import Renderer, VideoSink

            renderer = Renderer(config, initials=initials, total_frames=source.frame_count)
            sink = VideoSink(session_dir / "annotated.mp4", source.fps)
        total = source.frame_count or 0
        done = 0
        for frame, t in source.frames():
            state = engine.process_frame(frame, t)
            log.consume(state)
            if renderer is not None:
                sink.write(renderer.draw(frame, state))
            done += 1
            if progress is not None:
                progress(done, total)
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
        if sink is not None:
            annotated = sink.close(audio_source=video_path)

    return {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "annotated_video": str(annotated) if annotated else None,
        "shots_jsonl": str(session_dir / "shots.jsonl"),
        "summary": session_summary(session_dir / "shots.jsonl"),
    }
