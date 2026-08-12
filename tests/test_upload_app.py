"""Upload pipeline + interface tests, with scripted models (no weights)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from engine.config import EngineConfig
from tests.test_engine_synthetic import make_shot_positions, scripted_detector

cv2 = pytest.importorskip("cv2")


def write_synthetic_clip(path: Path, n_frames: int, fps: float = 60.0) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080)
    )
    frame = np.full((1080, 1920, 3), 80, dtype=np.uint8)
    for _ in range(n_frames):
        writer.write(frame)
    writer.release()


def counting_detector():
    """The scripted make, keyed by an internal frame counter (real sources
    hand the detector pixels, not indices)."""
    positions = make_shot_positions()
    inner = scripted_detector(positions)
    i = -1

    def detect(frame):
        nonlocal i
        i += 1
        return inner(i)

    return detect, max(positions) + 6


def test_run_upload_end_to_end(tmp_path):
    from app.pipeline import run_upload

    detector, n_frames = counting_detector()
    clip = tmp_path / "clip.mp4"
    write_synthetic_clip(clip, n_frames)

    seen = []
    result = run_upload(
        clip,
        detector=detector,
        config=EngineConfig.upload(),
        session_id="test-session",
        sessions_root=tmp_path / "sessions",
        progress=lambda done, total: seen.append((done, total)),
    )

    assert result["summary"]["attempts"] == 1
    assert result["summary"]["makes"] == 1
    assert Path(result["annotated_video"]).exists()
    lines = Path(result["shots_jsonl"]).read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["verdict"] == "make"
    assert seen and seen[-1][0] == n_frames  # progress reached the last frame

    out = cv2.VideoCapture(result["annotated_video"])
    assert out.isOpened() and int(out.get(cv2.CAP_PROP_FRAME_COUNT)) == n_frames
    out.release()


def test_upload_api_end_to_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app import server

    detector, n_frames = counting_detector()
    monkeypatch.setattr(
        server, "MODEL_FACTORY",
        lambda: {"config": EngineConfig.upload(), "detector": detector, "pose_model": None},
    )
    monkeypatch.setattr(server, "UPLOAD_ROOT", tmp_path / "uploads")
    monkeypatch.chdir(tmp_path)  # sessions land under the tmp dir

    clip = tmp_path / "clip.mp4"
    write_synthetic_clip(clip, n_frames)

    client = TestClient(server.app)
    assert "Drop a video" in client.get("/").text
    assert client.post(
        "/upload", files={"file": ("evil.exe", b"nope", "video/mp4")}
    ).status_code == 400

    with open(clip, "rb") as fh:
        r = client.post("/upload", files={"file": ("clip.mp4", fh, "video/mp4")})
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    for _ in range(300):
        status = client.get(f"/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert status["status"] == "done", status
    assert status["summary"]["makes"] == 1

    video = client.get(f"/jobs/{job_id}/video")
    assert video.status_code == 200 and len(video.content) > 10_000
    shots = client.get(f"/jobs/{job_id}/shots")
    assert shots.status_code == 200 and b'"verdict": "make"' in shots.content
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/jobs/nope/video").status_code == 404


def test_annotated_video_not_truncated_by_short_audio(tmp_path):
    """Regression: -shortest in the mux cut the video at the audio track's
    end (a 4s clip with 1s of audio lost 75% of its frames)."""
    import subprocess

    import imageio_ffmpeg

    from app.pipeline import run_upload

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clip = tmp_path / "clip.mp4"
    subprocess.run(
        [ffmpeg, "-y",
         "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=60:d=4",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)],
        check=True, capture_output=True,
    )
    detector, _ = counting_detector()
    result = run_upload(
        clip, detector=detector, config=EngineConfig.upload(),
        session_id="audio-test", sessions_root=tmp_path / "s",
    )
    out = cv2.VideoCapture(result["annotated_video"])
    n = int(out.get(cv2.CAP_PROP_FRAME_COUNT))
    out.release()
    assert n >= 235  # ~240 video frames must survive the 1s audio track


def test_back_to_back_runs_get_distinct_sessions(tmp_path):
    from app.pipeline import run_upload

    clip = tmp_path / "clip.mp4"
    d1, n = counting_detector()
    d2, _ = counting_detector()
    write_synthetic_clip(clip, n)
    kw = dict(config=EngineConfig.upload(), sessions_root=tmp_path / "s",
              render_video=False)
    r1 = run_upload(clip, detector=d1, **kw)
    r2 = run_upload(clip, detector=d2, **kw)
    assert r1["session_id"] != r2["session_id"]
    assert r1["summary"]["makes"] == r2["summary"]["makes"] == 1


def test_low_fps_warning_reaches_the_result(tmp_path):
    from app.pipeline import run_upload

    clip = tmp_path / "clip.mp4"
    d, n = counting_detector()
    write_synthetic_clip(clip, n, fps=30.0)
    result = run_upload(
        clip, detector=d, config=EngineConfig.upload(),
        sessions_root=tmp_path / "s", render_video=False,
    )
    assert any("fps" in w for w in result["warnings"])
