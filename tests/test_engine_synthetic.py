"""M0 acceptance: a scripted parabola of fake detections produces exactly one
RESOLVED make event — plus occlusion, gap, rattle, miss, and noise scenarios
around the same machinery.

No real models anywhere: the detector is a scripted callable, the "frame" fed
to the engine is just the frame index. The engine must not care.
"""

from __future__ import annotations

import math

import pytest

from engine.config import EngineConfig
from engine.engine import ShotEngine
from engine.types import Detection, ShotPhase, Verdict, VerdictConfidence

FPS = 60.0
RIM = (910.0, 280.0, 1010.0, 320.0)  # nominal 100px wide = 18in rim
# Widths alternate 100/101px via jitter; long-run median is 100.5px.
SCALE_PX_PER_M = 100.5 / 0.4572
ARC_A = 0.025952  # parabola curvature used by every scripted arc


def rim_det(i: int) -> Detection:
    j = float(i % 2)  # jitter the width so the calibration median does real work
    x1, y1, x2, y2 = RIM
    return Detection("rim", 0.95, (x1, y1, x2 + j, y2))


def ball_det(x: float, y: float) -> Detection:
    return Detection("ball", 0.9, (x - 15, y - 15, x + 15, y + 15))


def scripted_detector(positions: dict[int, tuple[float, float]], rim_until: int | None = None):
    def detect(frame_index):
        dets = [
            # A class the source dataset carries but the engine must ignore.
            Detection("made", 0.99, (100.0, 100.0, 160.0, 140.0)),
        ]
        if rim_until is None or frame_index < rim_until:
            dets.append(rim_det(frame_index))
        if frame_index in positions:
            dets.append(ball_det(*positions[frame_index]))
        return dets

    return detect


def run(positions: dict[int, tuple[float, float]], rim_until: int | None = None, extra: int = 6):
    n = max(positions) + extra  # run past the last detection to flush the tracker
    engine = ShotEngine(
        EngineConfig.upload(), detector=scripted_detector(positions, rim_until)
    )
    return [engine.process_frame(i, i / FPS) for i in range(n)]


def parabola(x: float, apex_x: float, apex_y: float) -> float:
    return ARC_A * (x - apex_x) ** 2 + apex_y


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


def make_shot_positions() -> dict[int, tuple[float, float]]:
    """The canonical scripted make: 10 frames in hands, then a clean swish."""
    positions = {i: (700.0, 900.0) for i in range(10)}
    positions |= arc_positions(10, 708.0, apex_x=870.0, apex_y=150.0, last_x=1000.0)
    return positions


def events_of(states):
    return [e for s in states for e in s.events]


# ---------------------------------------------------------------------- make


def test_scripted_parabola_produces_one_resolved_make():
    """The M0 acceptance criterion."""
    states = run(make_shot_positions())

    events = events_of(states)
    assert len(events) == 1
    e = events[0]
    assert e.verdict is Verdict.MAKE
    assert e.verdict_confidence is VerdictConfidence.CLEAN
    assert e.shot_id == 1
    # Deterministic script, deterministic frames: release is the first frame
    # of the upward streak; crossing lands on frame 40 and the first observed
    # below-rim frame is 41.
    assert e.frames == (10, 41)

    resolved = [s for s in states if s.phase is ShotPhase.RESOLVED]
    assert len(resolved) == 1
    assert resolved[0].frame_index == 41
    assert resolved[0].events == [e]
    assert states[-1].phase is ShotPhase.IDLE  # machine came back to rest


def test_full_phase_path_is_traversed_in_order():
    states = run(make_shot_positions())
    phases = [s.phase for s in states]

    first_rising = phases.index(ShotPhase.RISING)
    first_desc = phases.index(ShotPhase.DESCENDING)
    resolved_at = phases.index(ShotPhase.RESOLVED)
    assert first_rising == 12  # 3-frame streak that began on frame 10
    assert first_rising < first_desc < resolved_at == 41
    assert phases[resolved_at + 1] is ShotPhase.IDLE
    # No IDLE anywhere inside the attempt: one contiguous shot.
    assert all(p is not ShotPhase.IDLE for p in phases[first_rising:resolved_at])
    assert states[20].active_shot_id == 1


def test_make_metrics_are_computed():
    positions = make_shot_positions()
    e = events_of(run(positions))[0]

    # Scripted arc enters steeply: tangent at the crossing is ~75 deg.
    assert e.entry_angle_deg == pytest.approx(75.0, abs=5.0)

    # Rise from release (frame 10) to apex, converted at the session scale.
    release_y = positions[10][1]
    apex_y = min(y for _, y in positions.values())
    assert e.peak_height_m == pytest.approx((release_y - apex_y) / SCALE_PX_PER_M, rel=0.02)

    # Mean speed over the five post-release segments, at the session scale —
    # the exact value, so a multiply-vs-divide unit error cannot pass.
    seg_speeds = [
        math.hypot(positions[i + 1][0] - positions[i][0], positions[i + 1][1] - positions[i][1])
        * FPS
        for i in range(10, 15)
    ]
    expected = (sum(seg_speeds) / len(seg_speeds)) / SCALE_PX_PER_M
    assert e.release_velocity_ms == pytest.approx(expected, rel=0.02)

    assert e.t_release == pytest.approx(10 / FPS, abs=1e-6)
    # Pose-dependent fields must be null, never garbage, before M3.
    assert e.elbow_deg is None and e.knee_deg is None and e.release_height_m is None
    assert e.court_pos_m is None  # no homography in this session


def test_calibration_and_renderer_feeds():
    positions = make_shot_positions()
    states = run(positions)

    # Long-run median width is 100.5px (alternating 100/101) over 0.4572m.
    assert states[-1].scale_px_per_m == pytest.approx(SCALE_PX_PER_M, rel=1e-3)

    mid = states[30]  # near apex, scale is 31-sample median = 100px here
    assert mid.ball is not None and not mid.ball.interpolated
    bx, by = positions[30]
    rim_cx, rim_cy = 960.0, 300.0  # median rim box center at this point
    expected_px = math.hypot(bx - rim_cx, by - rim_cy)
    assert mid.distance_to_rim_m == pytest.approx(expected_px / mid.scale_px_per_m, rel=0.01)
    # At the apex speed is nearly all horizontal: ~480 px/s -> ~2.2 m/s.
    assert 1.0 < mid.current_speed_ms < 5.0
    # Mid-ascent (frame 20) the ball is fast; the band is tight enough to
    # catch a px<->m unit inversion (which lands at ~0.005 or ~5e5).
    assert 5.0 < states[20].current_speed_ms < 40.0
    assert 2 <= len(mid.trail) <= 20


def test_make_through_occlusion_reappears_below():
    """Crossing observed, then the net occludes the ball well past the
    tracker's 4-frame gap; it reappears below the rim -> make, occluded."""
    positions = {i: (700.0, 900.0) for i in range(10)}
    positions |= arc_positions(10, 708.0, apex_x=870.0, apex_y=150.0, last_x=948.0)
    # Last observed frame is 40 (x=948, just after the crossing). Frames
    # 41-45 are dark; the ball reappears on 46 well below the rim, in span.
    x46 = 996.0
    positions[46] = (x46, parabola(x46, 870.0, 150.0))
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    e = events[0]
    assert e.verdict is Verdict.MAKE
    assert e.verdict_confidence is VerdictConfidence.OCCLUDED
    assert e.frames == (10, 46)
    assert states[-1].phase is ShotPhase.IDLE


def test_make_survives_midflight_detection_gap():
    """Two dark frames on the way up (motion blur): still one clean make."""
    positions = make_shot_positions()
    del positions[20], positions[21]
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MAKE
    assert events[0].verdict_confidence is VerdictConfidence.CLEAN
    assert events[0].frames == (10, 41)


def test_rattle_in_is_one_make_not_two_events():
    """Ball crosses in-span, pops back out above the rim, then drops back
    through: exactly one event, a rattled make — not a miss plus a phantom
    second attempt."""
    positions = {i: (700.0, 900.0) for i in range(5)}
    positions |= arc_positions(5, 708.0, apex_x=870.0, apex_y=150.0, last_x=948.0)
    after = max(positions)  # frame 35, x=948, y~308: inside the cylinder
    bounce = [
        (951.0, 298.0), (953.0, 282.0), (955.0, 268.0), (957.0, 258.0), (959.0, 254.0),
        (961.0, 258.0), (963.0, 268.0), (965.0, 282.0), (967.0, 300.0), (969.0, 322.0),
        (971.0, 348.0),
    ]
    for k, pos in enumerate(bounce, start=1):
        positions[after + k] = pos
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    e = events[0]
    assert e.verdict is Verdict.MAKE
    assert e.verdict_confidence is VerdictConfidence.RATTLED
    assert e.shot_id == 1
    assert states[-1].phase is ShotPhase.IDLE


def test_make_resolves_from_median_rim_box_when_rim_detections_stop():
    """Rim detected only for the first 15 frames; the median box carries the
    whole shot (the rim doesn't move within a session)."""
    states = run(make_shot_positions(), rim_until=15)

    assert states[30].rim is None  # no live rim detection mid-flight
    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MAKE
    # 15 samples: eight 100px widths, seven 101px -> median 100.
    assert states[-1].scale_px_per_m == pytest.approx(100.0 / 0.4572, rel=1e-3)


def test_two_shots_one_engine():
    """A make, ball returns to hands, then a short miss: two events, ids 1
    and 2, no merged or phantom attempts."""
    positions = make_shot_positions()  # resolves MAKE at frame 41
    for i in range(60, 75):
        positions[i] = (630.0, 900.0)  # back in hands (continuous with the arc below)
    positions |= arc_positions(75, 638.0, apex_x=800.0, apex_y=150.0, last_x=940.0)
    states = run(positions)

    events = events_of(states)
    assert [e.shot_id for e in events] == [1, 2]
    assert [e.verdict for e in events] == [Verdict.MAKE, Verdict.MISS]
    assert states[-1].phase is ShotPhase.IDLE


# -------------------------------------------------------------------- misses


def test_short_airball_is_a_clean_miss():
    """Arc apexes early and falls in front of the rim: enters the rim
    neighborhood, never crosses the top edge in-span, exits below."""
    # Idle spot sits ON the launch parabola: the first flight frame moves
    # ~69px, inside the tracker gate, like a real continuous launch.
    positions = {i: (630.0, 900.0) for i in range(5)}
    positions |= arc_positions(5, 638.0, apex_x=800.0, apex_y=150.0, last_x=940.0)
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MISS
    assert events[0].verdict_confidence is VerdictConfidence.CLEAN
    assert events[0].entry_angle_deg is None  # no crossing -> no entry angle
    assert states[-1].phase is ShotPhase.IDLE


def test_rim_out_is_a_rattled_miss():
    """Ball crosses the rim's top edge in-span, pops out, and climbs away:
    resolved as a single rattled miss when it leaves the rim neighborhood —
    and the rebound must not be segmented as a second attempt."""
    positions = {i: (700.0, 900.0) for i in range(5)}
    positions |= arc_positions(5, 708.0, apex_x=870.0, apex_y=150.0, last_x=948.0)
    after = max(positions)  # frame 35: crossing frame
    bounce = [
        (952.0, 300.0), (956.0, 282.0), (960.0, 262.0), (964.0, 242.0), (968.0, 226.0),
        (972.0, 210.0), (976.0, 194.0), (980.0, 178.0), (984.0, 162.0), (988.0, 146.0),
    ]
    for k, pos in enumerate(bounce, start=1):
        positions[after + k] = pos
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MISS
    assert events[0].verdict_confidence is VerdictConfidence.RATTLED
    # Not resolved at the first pop-out frame (36-38); resolved once the pop
    # exceeds one rim width above the rim top (y < 180, frame 43).
    assert events[0].frames[1] == 43
    assert states[-1].phase is ShotPhase.IDLE


def test_high_bounce_rejection_is_one_rattled_miss():
    """Ball crosses in-span but gets launched high off the back rim (more
    than a rim width above the top): a rejection — one rattled miss, resolved
    at the pop-height threshold, and the fall-back-through afterwards must
    not create a second event."""
    positions = {i: (700.0, 900.0) for i in range(5)}
    positions |= arc_positions(5, 708.0, apex_x=870.0, apex_y=150.0, last_x=948.0)
    after = max(positions)  # frame 35: crossing frame
    flight = [
        (952.0, 296.0), (954.0, 250.0), (956.0, 200.0), (958.0, 160.0),  # rejection up
        (960.0, 120.0), (962.0, 140.0), (964.0, 170.0), (966.0, 220.0),  # back down...
        (968.0, 290.0), (970.0, 360.0),  # ...through the span again
    ]
    for k, pos in enumerate(flight, start=1):
        positions[after + k] = pos
    states = run(positions)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MISS
    assert events[0].verdict_confidence is VerdictConfidence.RATTLED
    assert events[0].frames[1] == 39  # pop passed one rim width above the top
    assert states[-1].phase is ShotPhase.IDLE


def test_reappearance_after_drop_cannot_arm_phantom_shot():
    """Ball occluded during a dribble, reappears higher and stationary (the
    pickup into the shot pocket): the re-seeded track has no measured
    velocity, so no phantom release can arm."""
    positions = {i: (700.0, 950.0) for i in range(10)}
    for i in range(30, 60):
        positions[i] = (700.0, 600.0)  # reappears 350px higher, then holds
    states = run(positions)

    assert events_of(states) == []
    assert all(s.phase is ShotPhase.IDLE for s in states)
    # And the re-seed frame must not report a fabricated speed.
    reseed = states[30]
    assert reseed.ball is not None and not reseed.ball.interpolated
    assert reseed.current_speed_ms is None


def test_false_positive_drizzle_cannot_wedge_the_machine():
    """Track lost near the rim, then a sporadic false 'ball' detection near
    the net every 20 frames: the hold must not stay open forever — the shot
    resolves at the max_shot_seconds cap and the machine returns to IDLE."""
    positions = {i: (700.0, 900.0) for i in range(5)}
    positions |= arc_positions(5, 708.0, apex_x=870.0, apex_y=150.0, last_x=852.0)
    for i in range(40, 401, 20):
        positions[i] = (940.0, 260.0)  # static near-net false positive
    states = run(positions, extra=30)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MISS
    assert events[0].verdict_confidence is VerdictConfidence.OCCLUDED
    cap_frames = int(EngineConfig.upload().max_shot_seconds * FPS)
    assert events[0].frames[1] <= events[0].frames[0] + cap_frames + 2
    assert states[-1].phase is ShotPhase.IDLE


def test_calibration_recovers_from_garbage_cold_start():
    """~1s of garbage rim detections while the camera settles, then the real
    rim: calibration must reset onto the real rim instead of locking the
    session to a phantom one (which would silently score nothing)."""
    garbage = Detection("rim", 0.45, (200.0, 800.0, 240.0, 830.0))
    positions = {i: (700.0, 900.0) for i in range(100, 110)}
    positions |= arc_positions(110, 708.0, apex_x=870.0, apex_y=150.0, last_x=1000.0)

    def detect(frame_index):
        dets = [garbage if frame_index < 55 else rim_det(frame_index)]
        if frame_index in positions:
            dets.append(ball_det(*positions[frame_index]))
        return dets

    engine = ShotEngine(EngineConfig.upload(), detector=detect)
    states = [engine.process_frame(i, i / FPS) for i in range(max(positions) + 6)]

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MAKE
    # Scale re-derived from the real rim, not the 40px garbage box.
    assert states[-1].scale_px_per_m == pytest.approx(SCALE_PX_PER_M, rel=0.02)


def test_stream_end_mid_shot_flushes_via_finalize():
    """Clip-per-shot footage cuts right at the outcome: a shot still open at
    end-of-stream must resolve through finalize(), not vanish."""
    positions = {i: (700.0, 900.0) for i in range(10)}
    # Ends right after the in-span crossing (frame 40); no below-rim frames.
    positions |= arc_positions(10, 708.0, apex_x=870.0, apex_y=150.0, last_x=948.0)
    engine = ShotEngine(EngineConfig.upload(), detector=scripted_detector(positions))
    states = [engine.process_frame(i, i / FPS) for i in range(max(positions) + 1)]

    assert events_of(states) == []  # nothing resolved while streaming
    flushed = engine.finalize()
    assert len(flushed) == 1
    assert flushed[0].verdict is Verdict.MISS  # crossing seen, landing never was
    assert flushed[0].verdict_confidence is VerdictConfidence.OCCLUDED
    assert engine.finalize() == []  # idempotent once idle


def test_track_lost_near_rim_is_occluded_miss_after_hold():
    """No crossing, ball vanishes inside the rim neighborhood and never
    reappears: miss, flagged occluded, resolved only after the hold window."""
    positions = {i: (700.0, 900.0) for i in range(5)}
    # Ball disappears right after entering the neighborhood, above the rim.
    positions |= arc_positions(5, 708.0, apex_x=870.0, apex_y=150.0, last_x=852.0)
    last = max(positions)  # x=852, y~158: inside neighborhood, pre-apex
    cfg = EngineConfig.upload()
    states = run(positions, extra=cfg.max_gap_frames + cfg.occlusion_hold_frames + 5)

    events = events_of(states)
    assert len(events) == 1
    assert events[0].verdict is Verdict.MISS
    assert events[0].verdict_confidence is VerdictConfidence.OCCLUDED
    # Held open through extrapolation + the reappear window before resolving.
    assert events[0].frames[1] >= last + cfg.occlusion_hold_frames
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


def test_no_rim_geometry_does_not_wedge_the_machine():
    """Rim never detected at all: a dribble candidate must still discard via
    the release-height rule instead of wedging in DESCENDING."""
    positions = {}
    for i in range(10):
        positions[i] = (700.0, 900.0 - 30.0 * i)
    for i in range(10, 25):
        positions[i] = (700.0, 600.0 + 30.0 * (i - 10))
    states = run(positions, rim_until=0)  # rim_until=0: no rim detections ever

    assert events_of(states) == []
    assert states[-1].phase is ShotPhase.IDLE


# ------------------------------------------------------------ event log shape


def test_shot_event_serializes_to_plan_schema(tmp_path):
    import json

    from stats.event_log import EventLog
    from stats.stats import session_summary

    states = run(make_shot_positions())

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

    # Append semantics within a run (plan 7), but a second run on the same
    # session must refuse rather than interleave colliding shot_ids.
    with pytest.raises(FileExistsError):
        EventLog(tmp_path / "session").__enter__()


# ------------------------------------------------------------- config guard


def test_modes_differ_only_in_model_size_config():
    up, live = EngineConfig.upload(), EngineConfig.live()
    assert (up.mode, live.mode) == ("upload", "live")
    # Every shot-logic threshold must be identical across modes.
    for field in (
        "max_gap_frames", "gate_base_px", "gate_speed_factor", "gate_gap_growth_px",
        "velocity_smoothing", "velocity_history_max_s", "reseed_speed_px_s",
        "release_consecutive_frames", "release_min_upward_speed_ms",
        "release_min_upward_speed_px_s",
        "rim_neighborhood_scale", "occlusion_frames_for_make",
        "occlusion_hold_frames", "span_tolerance_frac", "max_shot_seconds",
        "rattle_pop_max_rim_widths", "rim_diameter_m", "calibration_frames",
        "calibration_min_samples", "calibration_drift_frames",
    ):
        assert getattr(up, field) == getattr(live, field), field
