"""Engine configuration.

Upload mode and live mode share every threshold and rule; the ONLY thing that
may differ between the two presets is model-size configuration (architecture
rule). If you are tempted to fork any other field per mode, stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from models.registry import DETECTOR_V0_MODEL_ID


@dataclass(frozen=True)
class DetectorConfig:
    model_id: str
    size: str  # "small" (upload) | "nano" (live, trained only if M5 misses 25+ fps)
    ball_confidence: float = 0.4
    rim_confidence: float = 0.4


@dataclass(frozen=True)
class PoseConfig:
    enabled: bool = True  # M3: zero-shot RF-DETR Keypoint
    keypoint_confidence: float = 0.5  # below this at release -> form metrics are null
    every_n_frames: int = 1  # live mode may raise this at M5 (model-size/perf config)


@dataclass(frozen=True)
class EngineConfig:
    mode: str  # "upload" | "live"
    detector: DetectorConfig
    pose: PoseConfig = field(default_factory=PoseConfig)

    # Tracking (plan 6.1)
    max_gap_frames: int = 4  # extrapolate through detection gaps up to this many frames
    gate_base_px: float = 75.0  # motion-gate radius floor
    gate_speed_factor: float = 1.5  # gate widens with predicted per-frame travel
    gate_gap_growth_px: float = 50.0  # bounded extra gate per unmatched frame
    velocity_smoothing: float = 0.5  # EMA weight on the newest velocity sample
    velocity_history_max_s: float = 1.0  # older last-observation is too stale for velocity
    reseed_speed_px_s: float = 2500.0  # max plausible ball speed for post-drop re-seeding

    # Release detection (plan 5): sustained upward velocity over 3 consecutive
    # frames. The threshold lives in m/s (via calibrated scale) so raising the
    # ball into the shot pocket (~1-2 m/s) doesn't arm a release the way a
    # real launch (~7 m/s) does; the px value is the pre-calibration fallback.
    # (2.0: at 30 fps, motion blur plus EMA smoothing measured a real soft
    # launch at 2.2-2.7 m/s on session-2026-08-11 (IMG_7103). Windups reach
    # 2.0-3.6 m/s, so some arm early — that skews release timing/metrics on
    # those shots, and a ball raised to rim height and lowered again (pump
    # fake under the hoop) can log a phantom attempt: ball-only 2D cannot
    # tell it from a soft floater. M3's wrist-separation rule is the fix for
    # both; until then this is a documented wrong-verdict class near the rim.)
    release_consecutive_frames: int = 3
    release_min_upward_speed_ms: float = 2.0
    # With trusted wrist positions (M3 pose), the release rule becomes the
    # plan-5 original: ball separated from the wrist neighborhood AND rising.
    # Separation excludes ball-in-hand frames, so the velocity floor drops —
    # soft floaters arm, pump fakes never do.
    release_separation_m: float = 0.35
    release_separation_px: float = 60.0  # pre-calibration fallback
    release_min_upward_speed_with_pose_ms: float = 1.2
    # Pre-calibration fallback. 400 px/s sits at 1.8-2.9 m/s across the
    # scales measured on real sessions (140-220 px/m) — close to the m/s
    # floor instead of the far-too-permissive 200. The calibration hint
    # (last-known scale surviving drift resets) keeps this window rare.
    release_min_upward_speed_px_s: float = 400.0

    # Shot state machine / make-miss (plan 6.2, 6.3)
    rim_neighborhood_scale: float = 3.0  # rim box expanded by this factor = "rim neighborhood"
    occlusion_frames_for_make: int = 2  # occluded >= this after crossing, reappears below -> make
    crossing_bridge_max_frames: int = 4  # rim-top crossing may hide in a gap this long
    occlusion_hold_frames: int = 30  # track lost mid-shot: wait this long for reappearance
    span_tolerance_frac: float = 0.25  # lateral slack (x rim width) on the reappears-below check
    # Hard cap on one attempt (kills wedged holds). Generous because the clock
    # anchors at the armed release, which a windup can place seconds early —
    # a deliberate free-throw routine must not time out mid-flight. The check
    # runs after each frame's evidence, so same-frame resolutions always win.
    max_shot_seconds: float = 10.0
    rattle_pop_max_rim_widths: float = 1.0  # pop-out higher than this above the rim = rejection

    # Calibration recovery: live rim steadily contradicting the median box
    # for this many frames resets calibration (garbage cold-start / camera bump)
    calibration_drift_frames: int = 30

    # Calibration (plan 5): rim inner diameter 18in; median rim width over first 100 frames
    rim_diameter_m: float = 0.4572
    calibration_frames: int = 100
    calibration_min_samples: int = 10

    # Renderer support
    trail_length: int = 20

    @staticmethod
    def upload() -> "EngineConfig":
        return EngineConfig(
            mode="upload",
            detector=DetectorConfig(model_id=DETECTOR_V0_MODEL_ID, size="small"),
        )

    @staticmethod
    def live() -> "EngineConfig":
        # Same engine, same thresholds. Detector stays "small" until live mode
        # demonstrably misses 25+ fps at M5 — only then does a nano variant exist.
        base = EngineConfig.upload()
        return replace(base, mode="live", pose=replace(base.pose, every_n_frames=3))
