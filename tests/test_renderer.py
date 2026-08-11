"""Renderer smoke tests: composition must survive every FrameState shape the
engine produces (no pose, null metrics, events, idle), produce the fixed
canvas, and the sink must write a playable file."""

from __future__ import annotations

import numpy as np
import pytest

from engine.config import EngineConfig
from engine.engine import ShotEngine
from render import layout as L
from render.renderer import Renderer, VideoSink
from tests.test_engine_synthetic import (
    FPS,
    full_pose,
    make_shot_positions,
    scripted_detector,
)


def synthetic_states(with_pose: bool):
    positions = make_shot_positions()
    engine = ShotEngine(
        EngineConfig.upload(),
        detector=scripted_detector(positions),
        pose_model=(lambda f: full_pose((700.0, 900.0))) if with_pose else None,
    )
    return [engine.process_frame(i, i / FPS) for i in range(max(positions) + 6)]


@pytest.mark.parametrize("with_pose", [True, False])
def test_renderer_composes_every_frame(with_pose):
    states = synthetic_states(with_pose)
    renderer = Renderer(total_frames=len(states))
    frame = np.full((1080, 1920, 3), 90, dtype=np.uint8)
    for st in states:
        canvas = renderer.draw(frame, st)
        assert canvas.shape == (L.CANVAS_H, L.CANVAS_W, 3)
        assert canvas.dtype == np.uint8
    # The verdict landed and stayed on the HUD.
    assert renderer.hud.verdict == "make"
    assert renderer.hud.shot_speed_kmh is not None


def test_renderer_handles_odd_source_sizes():
    states = synthetic_states(False)[:5]
    renderer = Renderer()
    for shape in ((1018, 1860, 3), (720, 1280, 3), (1080, 608, 3)):
        frame = np.zeros(shape, dtype=np.uint8)
        canvas = renderer.draw(frame, states[0])
        assert canvas.shape == (L.CANVAS_H, L.CANVAS_W, 3)


def test_sink_writes_playable_file(tmp_path):
    import cv2

    states = synthetic_states(True)[:30]
    renderer = Renderer(total_frames=30)
    frame = np.full((1080, 1920, 3), 70, dtype=np.uint8)
    sink = VideoSink(tmp_path / "out.mp4", fps=60.0)
    for st in states:
        sink.write(renderer.draw(frame, st))
    out = sink.close()
    assert out.exists() and out.stat().st_size > 10_000
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 30
    cap.release()
