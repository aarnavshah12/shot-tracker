"""Shot state machine (plan 6.2, 6.3).

IDLE -> RISING -> DESCENDING -> RESOLVED(make|miss) -> IDLE

- IDLE -> RISING: release detected — sustained upward velocity over
  ``release_consecutive_frames`` observed frames. (M3 adds the wrist-
  separation half of the release rule; velocity is the whole rule until pose
  exists.)
- RISING -> DESCENDING: vertical velocity sign flip while the ball is above
  release height (or already inside the rim neighborhood — a bank shot's
  rebound off glass must not reset the machine).
- DESCENDING -> RESOLVED: make/miss crossing check, ball exits the rim
  neighborhood, or the trajectory ends.
- Anything that never rises toward the rim neighborhood is dribble noise: the
  candidate is discarded without an event and the machine returns to IDLE.

Make (6.3): centroid crosses the rim's top edge moving downward within the
rim's horizontal span, then is *observed* below the rim within the span — or
is occluded for >= 2 frames after crossing and reappears below. Everything
else is a miss, including rim-outs. ``verdict_confidence`` (clean | rattled |
occluded) is logged for auditing, never silently guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine import metrics
from engine.config import EngineConfig
from engine.types import BallTrack, ShotEvent, ShotPhase, Verdict, VerdictConfidence

Box = tuple[float, float, float, float]


@dataclass
class _ActiveShot:
    shot_id: int
    release_frame: int
    release_t: float
    release_y: float
    samples: list[metrics.Sample] = field(default_factory=list)
    entered_neighborhood: bool = False
    crossed: bool = False
    crossing_frame: Optional[int] = None
    crossing_x: Optional[float] = None
    post_cross_unobserved: int = 0
    rattled: bool = False


class ShotStateMachine:
    def __init__(self, config: EngineConfig):
        self._cfg = config
        self._phase = ShotPhase.IDLE
        self._next_shot_id = 1
        self._streak: list[metrics.Sample] = []  # consecutive rising observed samples
        self._shot: Optional[_ActiveShot] = None

    @property
    def phase(self) -> ShotPhase:
        return self._phase

    @property
    def active_shot_id(self) -> Optional[int]:
        return self._shot.shot_id if self._shot else None

    def update(
        self,
        ball: Optional[BallTrack],
        rim_box: Optional[Box],
        frame_index: int,
        t: float,
        px_per_m: Optional[float],
    ) -> Optional[ShotEvent]:
        """Advance one frame; returns a ShotEvent on the frame a verdict lands."""
        if self._phase is ShotPhase.RESOLVED:
            self._phase = ShotPhase.IDLE

        if self._phase is ShotPhase.IDLE:
            self._update_idle(ball, frame_index, t)
            return None
        if self._phase is ShotPhase.RISING:
            return self._update_rising(ball, rim_box, frame_index, t, px_per_m)
        return self._update_descending(ball, rim_box, frame_index, t, px_per_m)

    # ------------------------------------------------------------------ IDLE

    def _update_idle(self, ball: Optional[BallTrack], frame_index: int, t: float) -> None:
        rising = (
            ball is not None
            and not ball.interpolated
            and ball.vy <= -self._cfg.release_min_upward_speed_px_s
        )
        if not rising:
            self._streak = []
            return
        self._streak.append((frame_index, t, ball.x, ball.y, False))
        if len(self._streak) < self._cfg.release_consecutive_frames:
            return
        release = self._streak[0]
        self._shot = _ActiveShot(
            shot_id=self._next_shot_id,
            release_frame=release[0],
            release_t=release[1],
            release_y=release[3],
            samples=list(self._streak),
        )
        self._next_shot_id += 1
        self._streak = []
        self._phase = ShotPhase.RISING

    # ---------------------------------------------------------------- RISING

    def _update_rising(
        self,
        ball: Optional[BallTrack],
        rim_box: Optional[Box],
        frame_index: int,
        t: float,
        px_per_m: Optional[float],
    ) -> Optional[ShotEvent]:
        shot = self._shot
        if ball is None:  # trajectory ended mid-rise
            if shot.entered_neighborhood:
                return self._resolve(Verdict.MISS, VerdictConfidence.OCCLUDED, frame_index, px_per_m)
            self._discard()
            return None

        self._record(ball, rim_box, frame_index, t)

        if ball.vy > 0:  # vertical velocity sign flip
            if ball.y < shot.release_y or shot.entered_neighborhood:
                self._phase = ShotPhase.DESCENDING
            else:
                # Peaked at/below release height, never approached the rim:
                # dribble noise.
                self._discard()
        return None

    # ----------------------------------------------------------- DESCENDING

    def _update_descending(
        self,
        ball: Optional[BallTrack],
        rim_box: Optional[Box],
        frame_index: int,
        t: float,
        px_per_m: Optional[float],
    ) -> Optional[ShotEvent]:
        shot = self._shot

        if ball is None:  # track lost for good
            if not shot.entered_neighborhood:
                self._discard()
                return None
            # Crossed or not, the ball vanished near the rim before the make
            # rule could confirm an observation below it. Audit as occluded.
            return self._resolve(Verdict.MISS, VerdictConfidence.OCCLUDED, frame_index, px_per_m)

        prev = shot.samples[-1] if shot.samples else None
        self._record(ball, rim_box, frame_index, t)

        if rim_box is None:
            return None  # no rim geometry yet; keep accumulating trajectory
        rim_x1, rim_top, rim_x2, rim_bottom = rim_box

        # --- crossing check: top edge, downward, within horizontal span
        if prev is not None and ball.y > prev[3]:
            prev_y, prev_x = prev[3], prev[2]
            if prev_y < rim_top <= ball.y:
                frac = (rim_top - prev_y) / (ball.y - prev_y)
                x_cross = prev_x + (ball.x - prev_x) * frac
                if rim_x1 <= x_cross <= rim_x2:
                    if shot.crossed:
                        shot.rattled = True  # re-crossing after a bounce-out
                    else:
                        shot.crossed = True
                        shot.crossing_frame = frame_index
                        shot.crossing_x = x_cross

        if shot.crossed:
            return self._check_after_crossing(ball, rim_box, frame_index, px_per_m)
        return self._check_no_crossing(ball, rim_box, frame_index, px_per_m)

    def _check_after_crossing(
        self, ball: BallTrack, rim_box: Box, frame_index: int, px_per_m: Optional[float]
    ) -> Optional[ShotEvent]:
        shot = self._shot
        rim_x1, rim_top, rim_x2, rim_bottom = rim_box

        if ball.interpolated:
            shot.post_cross_unobserved += 1
            return None

        in_span = rim_x1 <= ball.x <= rim_x2
        if ball.y > rim_bottom and in_span:
            # Observed below the rim within the span: make. Confidence order:
            # a rattled make stays "rattled" even if the net also occluded it.
            if shot.rattled:
                conf = VerdictConfidence.RATTLED
            elif shot.post_cross_unobserved >= self._cfg.occlusion_frames_for_make:
                conf = VerdictConfidence.OCCLUDED
            else:
                conf = VerdictConfidence.CLEAN
            return self._resolve(Verdict.MAKE, conf, frame_index, px_per_m)

        if ball.y < rim_top and ball.vy < 0:
            # Popped back out above the rim: rim-out.
            return self._resolve(Verdict.MISS, VerdictConfidence.RATTLED, frame_index, px_per_m)
        if not in_span and ball.y <= rim_bottom:
            # Slid out sideways above the rim bottom: rim-out.
            return self._resolve(Verdict.MISS, VerdictConfidence.RATTLED, frame_index, px_per_m)
        if ball.y > rim_bottom and not in_span:
            # Below rim height but outside the span — never seen through the
            # cylinder. (Oblique-camera airball guard: span + below must BOTH
            # pass for a make.)
            return self._resolve(Verdict.MISS, VerdictConfidence.RATTLED, frame_index, px_per_m)

        shot.rattled = shot.rattled or ball.vy < 0  # bouncing on the rim
        return None

    def _check_no_crossing(
        self, ball: BallTrack, rim_box: Box, frame_index: int, px_per_m: Optional[float]
    ) -> Optional[ShotEvent]:
        shot = self._shot
        inside = self._in_neighborhood(ball, rim_box)
        if shot.entered_neighborhood and not inside:
            # Approached the rim, left it without ever crossing the top edge.
            return self._resolve(Verdict.MISS, VerdictConfidence.CLEAN, frame_index, px_per_m)
        if not shot.entered_neighborhood and ball.y >= shot.release_y:
            # Fell back to release height without approaching the rim.
            self._discard()
        return None

    # ------------------------------------------------------------- plumbing

    def _record(
        self, ball: BallTrack, rim_box: Optional[Box], frame_index: int, t: float
    ) -> None:
        shot = self._shot
        shot.samples.append((frame_index, t, ball.x, ball.y, ball.interpolated))
        if rim_box is not None and self._in_neighborhood(ball, rim_box):
            shot.entered_neighborhood = True

    def _in_neighborhood(self, ball: BallTrack, rim_box: Box) -> bool:
        x1, y1, x2, y2 = rim_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = self._cfg.rim_neighborhood_scale * max(x2 - x1, y2 - y1) / 2
        return abs(ball.x - cx) <= half and abs(ball.y - cy) <= half

    def _discard(self) -> None:
        self._shot = None
        self._next_shot_id -= 1  # candidate was noise; don't burn the id
        self._phase = ShotPhase.IDLE

    def _resolve(
        self,
        verdict: Verdict,
        confidence: VerdictConfidence,
        frame_index: int,
        px_per_m: Optional[float],
    ) -> ShotEvent:
        shot = self._shot
        event = ShotEvent(
            shot_id=shot.shot_id,
            t_release=shot.release_t,
            verdict=verdict,
            verdict_confidence=confidence,
            frames=(shot.release_frame, frame_index),
            entry_angle_deg=self._entry_angle(shot),
            peak_height_m=metrics.peak_height_m(shot.samples, shot.release_y, px_per_m),
            release_velocity_ms=metrics.release_velocity_ms(
                shot.samples, shot.release_frame, px_per_m
            ),
        )
        self._shot = None
        self._phase = ShotPhase.RESOLVED
        return event

    @staticmethod
    def _entry_angle(shot: _ActiveShot) -> Optional[float]:
        if not shot.crossed or shot.crossing_x is None:
            return None
        upto = [s for s in shot.samples if s[0] <= shot.crossing_frame]
        observed = [s for s in upto if not s[4]]
        if not observed:
            return None
        apex_frame = min(observed, key=lambda s: s[3])[0]
        arc = [s for s in upto if s[0] >= apex_frame]
        return metrics.entry_angle_deg(arc, shot.crossing_x)
