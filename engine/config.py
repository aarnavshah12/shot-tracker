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
    enabled: bool = False  # flipped on at M3
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
    velocity_smoothing: float = 0.5  # EMA weight on the newest velocity sample
    velocity_history_max_s: float = 1.0  # older last-observation is too stale for velocity

    # Release detection (plan 5): sustained upward velocity over 3 consecutive frames
    release_consecutive_frames: int = 3
    release_min_upward_speed_px_s: float = 200.0

    # Shot state machine / make-miss (plan 6.2, 6.3)
    rim_neighborhood_scale: float = 3.0  # rim box expanded by this factor = "rim neighborhood"
    occlusion_frames_for_make: int = 2  # occluded >= this after crossing, reappears below -> make
    occlusion_hold_frames: int = 30  # track lost mid-shot: wait this long for reappearance
    span_tolerance_frac: float = 0.25  # lateral slack (x rim width) on the reappears-below check

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
