"""Upload-mode pipeline: VideoFileSource -> ShotEngine -> EventLog + Renderer.

This is wiring only. Shot logic lives in the engine; drawing in the renderer;
file I/O here and in the event log/sink. The upload interface (app/server.py)
calls run_upload with a progress callback.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from engine.config import EngineConfig
from engine.engine import Detector, PoseModel, ShotEngine
from stats.event_log import EventLog
from stats.stats import session_summary

# Anchored to the repo, not the process cwd: a server launched from anywhere
# must not scatter sessions across the filesystem (or 500 on cwd='/').
DEFAULT_SESSIONS_ROOT = Path(__file__).resolve().parent.parent / "sessions"


def run_upload(
    video_path: str | Path,
    detector: Detector,
    pose_model: Optional[PoseModel] = None,
    config: Optional[EngineConfig] = None,
    session_id: Optional[str] = None,
    sessions_root: str | Path = DEFAULT_SESSIONS_ROOT,
    render_video: bool = True,
    initials: str = "AA",
    progress: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Process one clip offline; returns paths plus the session summary."""
    from sources.video_file import VideoFileSource  # deferred: needs OpenCV

    config = config or EngineConfig.upload()
    if session_id is None:
        # Second resolution alone collides on back-to-back runs and either
        # crashes EventLog or silently clobbers the previous session.
        session_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    session_dir = Path(sessions_root) / session_id

    engine = ShotEngine(config, detector=detector, pose_model=pose_model)
    annotated: Optional[Path] = None
    run_warnings: list[str] = []

    with VideoFileSource(video_path) as source, EventLog(session_dir) as log:
        run_warnings.extend(source.warnings)
        renderer = sink = None
        if render_video:
            from render.renderer import Renderer, VideoSink

            renderer = Renderer(config, initials=initials, total_frames=source.frame_count)
            sink = VideoSink(session_dir / "annotated.mp4", source.fps)
        total = source.frame_count
        done = 0
        try:
            for frame, t in source.frames():
                state = engine.process_frame(frame, t)
                log.consume(state)
                if renderer is not None:
                    sink.write(renderer.draw(frame, state))
                done += 1
                if progress is not None:
                    progress(done, total)
        except BaseException:
            if sink is not None:
                sink.abort()  # no orphaned raw renders on failure
            raise

        if done == 0:
            if sink is not None:
                sink.abort()
            raise ValueError(
                "no frames could be decoded — the clip is corrupt or unsupported"
            )
        if total and done < total * 0.9:
            run_warnings.append(
                f"decoded only {done} of {total} expected frames — the clip "
                "may be truncated; stats cover the decoded part only"
            )

        # A shot still open when the clip ends is a real trajectory end.
        log.consume_events(engine.finalize())
        log.write_session_metadata(
            {
                "session_id": session_id,
                "source": str(video_path),
                "fps": source.fps,
                "mode": config.mode,
                "calibration": engine.calibration.to_metadata(),
                "warnings": run_warnings,
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
        "decoded_frames": done,
        "expected_frames": total,
        "warnings": run_warnings,
    }
