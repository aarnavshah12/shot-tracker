"""M0 acceptance: a scripted parabola of fake detections produces exactly one
RESOLVED make event — plus miss/noise scenarios around the same machinery.

No real models anywhere: the detector is a scripted callable, the "frame" fed
to the engine is just the frame index. The engine must not care.
"""

from __future__ import annotations

import pytest

from engine.config import EngineConfig
from engine.engine import ShotEngine
from engine.types import Detection, ShotPhase, Verdict, VerdictConfidence

FPS = 60.0
RIM = (910.0, 280.0, 1010.0, 320.0)  # 100px wide = 18in rim -> ~218.7 px/m
RIM_NEIGHBORHOOD = 150.0  # 3 * max(rim w, rim h) / 2, from config defaults


def rim_det(i: int) -> Detection:
    j = float(i % 2)  # 1px jitter so the calibration median does real work
    x1, y1, x2, y2 = RIM
    return Detection("rim", 0.95, (x1 + j, y1, x2 + j, y2))


def ball_det(x: float, y: float) -> Detection:
    return Detection("ball", 0.9, (x - 15, y - 15, x + 15, y + 15))


def scripted_detector(positions: dict[int, tuple[float, float]]):
    def detect(frame_index):
        dets = [
            rim_det(frame_index),
            # A class the source dataset carries but the engine must ignore.
            Detection("made", 0.99, (100.0, 100.0, 160.0, 140.0)),
        ]
        if frame_index in positions:
            dets.append(ball_det(*positions[frame_index]))
        return dets

    return detect


def run(positions: dict[int, tuple[float, float]]):
    n = max(positions) + 6  # run past the last detection to flush the tracker
    engine = ShotEngine(EngineConfig.upload(), detector=scripted_detector(positions))
    return [engine.process_frame(i, i / FPS) for i in range(n)]


def parabola(x: float, apex_x: float, apex_y: float, a: float = 0.025952) -> float:
    return a * (x - apex_x) ** 2 + apex_y


def arc_positions(
    start_frame: int, x0: float, apex_x: float, apex_y: float, last_x: float
) -> dict[int, tuple[float, float]]:
    """Ball moving right at 8 px/frame along a parabola (y down)."""
    positions = {}
    i, x = start_frame, x0
    while x <= last_x:
        positions[i] = (x, parabola(x, apex_x, apex_y))
        i, x = i + 1, x + 8.0
    return positions


def events_of(states):
    return [e for s in states for e in s.events]


# ---------------------------------------------------------------------- make


def test_scripted_parabola_produces_one_resolved_make():
    """The M0 acceptance criterion."""
    positions = {i: (700.0, 900.0) for i in range(10)}  # ball in hands
    positions |= arc_positions(10, 708.0, apex_x=870.0, apex_y=150.0, last_x=1000.0)
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    e = events[0]
    assert e.verdict is Verdict.MAKE
    assert e.verdict_confidence is VerdictConfidence.CLEAN
    assert e.shot_id == 1
    assert e.frames[0] == 10  # release: first frame of the upward streak
    assert 39 <= e.frames[1] <= 43  # resolved just after the rim crossing

    resolved = [s for s in states if s.phase is ShotPhase.RESOLVED]
    assert len(resolved) == 1
    assert resolved[0].events == [e]
    assert states[-1].phase is ShotPhase.IDLE  # machine came back to rest


def test_make_metrics_are_computed():
    positions = {i: (700.0, 900.0) for i in range(10)}
    positions |= arc_positions(10, 708.0, apex_x=870.0, apex_y=150.0, last_x=1000.0)
    e = events_of(run(positions))[0]

    # Scripted arc enters steeply: tangent at the crossing is ~75 deg.
    assert e.entry_angle_deg == pytest.approx(75.0, abs=5.0)
    # Rise from release (y=831) to apex (y=150) at ~218.7 px/m.
    assert e.peak_height_m == pytest.approx(3.11, abs=0.15)
    assert e.release_velocity_ms is not None and e.release_velocity_ms > 0
    assert e.t_release == pytest.approx(10 / FPS, abs=1e-6)
    # Pose-dependent fields must be null, never garbage, before M3.
    assert e.elbow_deg is None and e.knee_deg is None and e.release_height_m is None
    assert e.court_pos_m is None  # no homography in this session


def test_calibration_and_renderer_feeds():
    positions = {i: (700.0, 900.0) for i in range(10)}
    positions |= arc_positions(10, 708.0, apex_x=870.0, apex_y=150.0, last_x=1000.0)
    states = run(positions)

    # Median rim width 100-101px over 0.4572m.
    assert states[30].scale_px_per_m == pytest.approx(100.5 / 0.4572, rel=0.02)
    mid_flight = states[30]
    assert mid_flight.ball is not None
    assert mid_flight.distance_to_rim_m is not None and mid_flight.distance_to_rim_m > 0
    assert mid_flight.current_speed_ms is not None and mid_flight.current_speed_ms > 0
    assert 2 <= len(mid_flight.trail) <= 20


# -------------------------------------------------------------------- misses


def test_short_airball_is_a_clean_miss():
    """Arc apexes early and falls in front of the rim: enters the rim
    neighborhood, never crosses the top edge in-span, exits below."""
    positions = {i: (632.0, 900.0) for i in range(5)}
    positions |= arc_positions(5, 640.0, apex_x=800.0, apex_y=150.0, last_x=940.0)
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MISS
    assert events[0].verdict_confidence is VerdictConfidence.CLEAN
    assert events[0].entry_angle_deg is None  # no crossing -> no entry angle
    assert states[-1].phase is ShotPhase.IDLE


def test_rim_out_is_a_rattled_miss():
    """Ball crosses the rim's top edge in-span, sits in the cylinder, then
    pops back out above the rim: miss, flagged rattled."""
    positions = {i: (700.0, 900.0) for i in range(5)}
    # In-parabola until just past the crossing (x=948 -> y~308, inside cylinder)
    positions |= arc_positions(5, 708.0, apex_x=870.0, apex_y=150.0, last_x=948.0)
    after = max(positions)
    bounce = [(952.0, 300.0), (956.0, 282.0), (960.0, 262.0), (964.0, 242.0), (968.0, 226.0)]
    for k, pos in enumerate(bounce, start=1):
        positions[after + k] = pos
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MISS
    assert events[0].verdict_confidence is VerdictConfidence.RATTLED
    # The rebound must not be segmented as a second attempt.
    assert states[-1].phase is ShotPhase.IDLE


# --------------------------------------------------------------------- noise


def test_dribble_bounce_emits_nothing():
    """Ball rises fast but peaks far below the rim neighborhood: dribble
    noise, no attempt logged (plan 6.2)."""
    positions = {}
    for i in range(10):
        positions[i] = (700.0, 900.0 - 30.0 * i)  # up to y=630
    for i in range(10, 21):
        positions[i] = (700.0, 600.0 + 30.0 * (i - 10))  # back down

    states = run(positions)
    assert events_of(states) == []
    assert all(s.phase is not ShotPhase.RESOLVED for s in states)
    assert states[-1].phase is ShotPhase.IDLE


def test_stationary_ball_stays_idle():
    positions = {i: (700.0, 900.0) for i in range(30)}
    states = run(positions)
    assert events_of(states) == []
    assert all(s.phase is ShotPhase.IDLE for s in states)


# ------------------------------------------------------------ event log shape


def test_shot_event_serializes_to_plan_schema(tmp_path):
    import json

    from stats.event_log import EventLog
    from stats.stats import session_summary

    positions = {i: (700.0, 900.0) for i in range(10)}
    positions |= arc_positions(10, 708.0, apex_x=870.0, apex_y=150.0, last_x=1000.0)
    states = run(positions)

    with EventLog(tmp_path / "session") as log:
        for s in states:
            log.consume(s)

    lines = (tmp_path / "session" / "shots.jsonl").read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert set(row) == {
        "shot_id", "t_release", "verdict", "verdict_confidence",
        "entry_angle_deg", "peak_height_m", "release_velocity_ms",
        "elbow_deg", "knee_deg", "shoulder_hip_deg",
        "release_height_m", "court_pos_m", "frames",
    }
    assert row["verdict"] == "make"

    summary = session_summary(tmp_path / "session" / "shots.jsonl")
    assert summary["attempts"] == 1
    assert summary["fg_pct"] == 100.0
    assert summary["streaks"] == {"current": 1, "best": 1}


# ------------------------------------------------------------- config guard


def test_modes_differ_only_in_model_size_config():
    up, live = EngineConfig.upload(), EngineConfig.live()
    assert (up.mode, live.mode) == ("upload", "live")
    # Every shot-logic threshold must be identical across modes.
    for field in (
        "max_gap_frames", "gate_base_px", "gate_speed_factor", "velocity_smoothing",
        "release_consecutive_frames", "release_min_upward_speed_px_s",
        "rim_neighborhood_scale", "occlusion_frames_for_make",
        "rim_diameter_m", "calibration_frames", "calibration_min_samples",
    ):
        assert getattr(up, field) == getattr(live, field), field
