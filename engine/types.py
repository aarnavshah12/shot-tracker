"""Core datatypes shared across the engine, renderer, and event log.

Coordinate convention: image space, origin top-left, y grows downward.
"Upward" motion therefore means negative vy. Velocities are px/second.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ShotPhase(Enum):
    IDLE = "idle"
    RISING = "rising"
    DESCENDING = "descending"
    RESOLVED = "resolved"  # transient: held for the single frame a verdict lands on


class Verdict(Enum):
    MAKE = "make"
    MISS = "miss"


class VerdictConfidence(Enum):
    CLEAN = "clean"
    RATTLED = "rattled"
    OCCLUDED = "occluded"


@dataclass(frozen=True)
class Detection:
    """One detector output box, (x1, y1, x2, y2) in pixels."""

    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass(frozen=True)
class BallTrack:
    """The tracked ball for one frame.

    ``interpolated`` marks positions the tracker extrapolated through a
    detection gap rather than observed; verdict rules that require the ball to
    be *seen* must check this flag. ``velocity_valid`` is False on a freshly
    seeded track with no displacement history — vx/vy are 0.0 then and must
    not be reported as a measured speed.
    """

    x: float
    y: float
    vx: float
    vy: float
    interpolated: bool
    bbox: Optional[tuple[float, float, float, float]] = None
    velocity_valid: bool = True


@dataclass(frozen=True)
class PoseState:
    """COCO-17 keypoints for the shooter; filled in at M3, None before that.

    keypoints maps name -> (x, y, confidence).
    """

    keypoints: dict[str, tuple[float, float, float]]


@dataclass
class ShotEvent:
    """One resolved shot attempt; serialized as a line of shots.jsonl.

    Metric fields are None when they cannot be computed honestly (no scale,
    low keypoint confidence, no homography) — never garbage values.
    """

    shot_id: int
    t_release: float
    verdict: Verdict
    verdict_confidence: VerdictConfidence
    frames: tuple[int, int]  # (release frame, resolution frame)
    entry_angle_deg: Optional[float] = None
    peak_height_m: Optional[float] = None  # apex rise above the release point
    release_velocity_ms: Optional[float] = None
    elbow_deg: Optional[float] = None
    knee_deg: Optional[float] = None
    shoulder_hip_deg: Optional[float] = None
    release_height_m: Optional[float] = None
    court_pos_m: Optional[tuple[float, float]] = None

    def to_json_dict(self) -> dict:
        return {
            "shot_id": self.shot_id,
            "t_release": round(self.t_release, 3),
            "verdict": self.verdict.value,
            "verdict_confidence": self.verdict_confidence.value,
            "entry_angle_deg": _round(self.entry_angle_deg),
            "peak_height_m": _round(self.peak_height_m),
            "release_velocity_ms": _round(self.release_velocity_ms),
            "elbow_deg": _round(self.elbow_deg),
            "knee_deg": _round(self.knee_deg),
            "shoulder_hip_deg": _round(self.shoulder_hip_deg),
            "release_height_m": _round(self.release_height_m),
            "court_pos_m": list(self.court_pos_m) if self.court_pos_m else None,
            "frames": list(self.frames),
        }


def _round(v: Optional[float], digits: int = 1) -> Optional[float]:
    return None if v is None else round(v, digits)


@dataclass
class FrameState:
    """Everything the engine knows after one frame; consumed by the renderer
    and the event log. The engine emits this and nothing else."""

    frame_index: int
    t: float
    phase: ShotPhase
    ball: Optional[BallTrack] = None
    rim: Optional[Detection] = None
    pose: Optional[PoseState] = None
    trail: list[tuple[float, float]] = field(default_factory=list)  # last ~20 centroids
    events: list[ShotEvent] = field(default_factory=list)  # shots resolved on this frame
    active_shot_id: Optional[int] = None
    scale_px_per_m: Optional[float] = None
    distance_to_rim_m: Optional[float] = None
    current_speed_ms: Optional[float] = None  # ball speed this frame, for the renderer
