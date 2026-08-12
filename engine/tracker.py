"""Single-ball tracker (plan 6.1).

Centroid matching with a constant-velocity motion gate. One shooter, one ball:
each frame we pick the detection nearest the predicted position inside the
gate (highest confidence to seed a new track). Through detection gaps of up to
``max_gap_frames`` we emit constant-velocity extrapolations flagged
``interpolated=True``; longer gaps drop the track.

Rules that keep occlusions honest:
- the gate grows by a bounded amount per unmatched frame — enough to re-catch
  a rim deflection, not enough to adopt a far-away false detection;
- while the track is alive, velocity is differenced against the last
  *observed* position over the real elapsed time, never against a drifted
  ghost, so in-gap re-acquisition can't produce a velocity spike that fakes a
  rim-out;
- velocity history dies with the track: a re-seeded track starts with
  ``velocity_valid=False`` (a position jump across a drop is not a measured
  speed, and must not be able to fake a rising streak);
- re-seeding after a drop is gated too: within ``velocity_history_max_s`` of
  the drop point, only detections a plausible ball flight away from where the
  track went dark are accepted — no teleports onto false positives across the
  frame. Older than that, any ball is a fresh start.

If identity flicker becomes a problem on real footage, swap in ByteTrack via
supervision behind this same interface.
"""

from __future__ import annotations

import math
from typing import Optional

from engine.config import EngineConfig
from engine.types import BallTrack, Detection

# Frame-to-frame dt beyond this is a timestamp glitch, not real travel time;
# don't let it inflate the motion gate.
_MAX_STEP_S = 0.25


class BallTracker:
    def __init__(self, config: EngineConfig):
        self._cfg = config
        self._track: Optional[BallTrack] = None
        self._last_t: Optional[float] = None
        self._gap = 0  # consecutive frames extrapolated
        self._last_obs: Optional[tuple[float, float, float]] = None  # (x, y, t), live track only
        self._reseed_anchor: Optional[tuple[float, float, float]] = None  # where track went dark

    @property
    def gap_frames(self) -> int:
        return self._gap

    def update(self, detections: list[Detection], t: float) -> Optional[BallTrack]:
        dt = None if self._last_t is None else t - self._last_t
        if dt is not None and dt <= 0:
            # Non-monotonic timestamp; treat as unusable for velocity.
            dt = None

        match = self._match(detections, dt, t)

        if match is not None:
            self._track = self._observed(match, t)
            self._gap = 0
            self._last_obs = (match.cx, match.cy, t)
            self._reseed_anchor = None
        elif self._track is not None and self._gap < self._cfg.max_gap_frames:
            self._track = self._extrapolate(self._track, dt)
            self._gap += 1
        else:
            if self._track is not None:
                self._reseed_anchor = self._last_obs  # remember where we lost it
            self._track = None
            self._gap = 0
            self._last_obs = None  # velocity history dies with the track

        self._last_t = t
        return self._track

    def _match(
        self, detections: list[Detection], dt: Optional[float], t: float
    ) -> Optional[Detection]:
        if not detections:
            return None
        if self._track is None:
            return self._reseed(detections, t)

        px, py = self._predict(self._track, dt)
        speed = math.hypot(self._track.vx, self._track.vy)
        step = min(dt, _MAX_STEP_S) if dt else 0.0
        gate = (
            self._cfg.gate_base_px
            + self._cfg.gate_speed_factor * speed * step
            + self._cfg.gate_gap_growth_px * self._gap
        )

        best, best_dist = None, math.inf
        for d in detections:
            dist = math.hypot(d.cx - px, d.cy - py)
            if dist <= gate and dist < best_dist:
                best, best_dist = d, dist
        return best

    def _reseed(self, detections: list[Detection], t: float) -> Optional[Detection]:
        if self._reseed_anchor is not None:
            ax, ay, at = self._reseed_anchor
            elapsed = t - at
            if elapsed <= 0 or elapsed > self._cfg.velocity_history_max_s:
                self._reseed_anchor = None  # stale: any ball is a fresh start
            else:
                radius = self._cfg.gate_base_px + self._cfg.reseed_speed_px_s * elapsed
                near = [
                    d for d in detections if math.hypot(d.cx - ax, d.cy - ay) <= radius
                ]
                if not near:
                    return None  # no plausible continuation; don't teleport
                return max(near, key=lambda d: d.confidence)
        return max(detections, key=lambda d: d.confidence)

    def _observed(self, det: Detection, t: float) -> BallTrack:
        prev = self._track
        if prev is not None and self._last_obs is not None:
            ox, oy, ot = self._last_obs
            span = t - ot
            if 0 < span <= self._cfg.velocity_history_max_s:
                raw_vx = (det.cx - ox) / span
                raw_vy = (det.cy - oy) / span
                if prev.velocity_valid and self._gap == 0:
                    a = self._cfg.velocity_smoothing
                    vx = a * raw_vx + (1 - a) * prev.vx
                    vy = a * raw_vy + (1 - a) * prev.vy
                else:
                    # Re-acquisition within a live gap: trust the raw
                    # displacement over the ghost-influenced estimate.
                    vx, vy = raw_vx, raw_vy
                return BallTrack(
                    x=det.cx, y=det.cy, vx=vx, vy=vy,
                    interpolated=False, bbox=det.bbox, velocity_valid=True,
                    confidence=det.confidence,
                )
        return BallTrack(
            x=det.cx, y=det.cy, vx=0.0, vy=0.0,
            interpolated=False, bbox=det.bbox, velocity_valid=False,
            confidence=det.confidence,
        )

    def _extrapolate(self, track: BallTrack, dt: Optional[float]) -> BallTrack:
        step = dt if dt else 0.0
        return BallTrack(
            x=track.x + track.vx * step,
            y=track.y + track.vy * step,
            vx=track.vx,
            vy=track.vy,
            interpolated=True,
            bbox=None,
            velocity_valid=track.velocity_valid,
        )

    @staticmethod
    def _predict(track: BallTrack, dt: Optional[float]) -> tuple[float, float]:
        step = min(dt, _MAX_STEP_S) if dt else 0.0
        return track.x + track.vx * step, track.y + track.vy * step
