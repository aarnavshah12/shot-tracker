"""Shot state machine (plan 6.2, 6.3).

IDLE -> RISING -> DESCENDING -> RESOLVED(make|miss) -> IDLE

- IDLE -> RISING: release detected — sustained upward velocity over
  ``release_consecutive_frames`` observed frames, and never while the ball is
  inside the rim neighborhood (residual rim-bounce flight is not a release;
  M3 adds the wrist-separation half of the rule).
- RISING -> DESCENDING: vertical velocity sign flip while the ball is above
  release height (or already inside the rim neighborhood — a bank shot's
  rebound off glass must not reset the machine), or a positional rim-top
  crossing observed before the smoothed velocity has flipped (flat arcs at
  rim height live in that 1-2 frame EMA lag window).
- DESCENDING -> RESOLVED: make/miss rules below, ball exits the rim
  neighborhood, or the trajectory ends after an ``occlusion_hold_frames``
  reappear window (a net occlusion routinely outlives the tracker's 4-frame
  extrapolation, so track loss alone must not resolve the shot).
- Anything that never rises toward the rim neighborhood is dribble noise: the
  candidate is discarded without an event and the machine returns to IDLE
  (plan 6.2; revisit the neighborhood size at M2 if real airballs get eaten).

Make (6.3): centroid crosses the rim's top edge moving downward within the
rim's horizontal span, then is *observed* below the rim near the span (a
``span_tolerance_frac`` margin absorbs lateral net drift) — including via the
occluded-crossing path where the passage itself fell in a detection gap.
Everything else is a miss, including rim-outs — which resolve only when the
ball leaves the rim neighborhood, never at the first pop-out frame, so one
rattle produces one event. ``verdict_confidence`` (clean | rattled |
occluded) is logged for auditing, never silently guessed.

Crossing geometry uses the median rim box supplied by the engine once
calibration has one (the rim doesn't move within a session), so per-frame
detection jitter can't fake re-crossings.
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
    crossing_occluded: bool = False  # the crossing segment spanned unobserved frames
    post_cross_unobserved: int = 0
    unseen_streak: int = 0  # consecutive frames with no track at all
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
            self._update_idle(ball, rim_box, frame_index, t)
            return None
        return self._update_airborne(ball, rim_box, frame_index, t, px_per_m)

    # ------------------------------------------------------------------ IDLE

    def _update_idle(
        self, ball: Optional[BallTrack], rim_box: Optional[Box], frame_index: int, t: float
    ) -> None:
        rising = (
            ball is not None
            and not ball.interpolated
            and ball.vy <= -self._cfg.release_min_upward_speed_px_s
        )
        if rising and rim_box is not None and self._in_neighborhood(ball, rim_box):
            # Upward flight inside the rim neighborhood is a bounce off the
            # rim, not a release: a shooter's hands are never up there.
            rising = False
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

    # -------------------------------------------------------------- airborne

    def _update_airborne(
        self,
        ball: Optional[BallTrack],
        rim_box: Optional[Box],
        frame_index: int,
        t: float,
        px_per_m: Optional[float],
    ) -> Optional[ShotEvent]:
        shot = self._shot

        if ball is None:
            # Track lost. Hold the shot open for a reappear window — net/body
            # occlusion at the rim routinely outlives the tracker's gap limit.
            shot.unseen_streak += 1
            if shot.unseen_streak <= self._cfg.occlusion_hold_frames:
                return None
            if not shot.entered_neighborhood:
                self._discard()  # vanished without ever approaching the rim
                return None
            return self._resolve(Verdict.MISS, VerdictConfidence.OCCLUDED, frame_index, px_per_m)
        shot.unseen_streak = 0

        prev = shot.samples[-1] if shot.samples else None
        self._record(ball, rim_box, frame_index, t)
        if rim_box is not None:
            self._detect_crossing(prev, ball, rim_box, frame_index)

        if self._phase is ShotPhase.RISING:
            if shot.crossed:
                # Positional descent through the rim top observed while the
                # smoothed velocity still says "rising": we're descending.
                self._phase = ShotPhase.DESCENDING
            elif ball.vy > 0:  # vertical velocity sign flip
                if ball.y < shot.release_y or shot.entered_neighborhood:
                    self._phase = ShotPhase.DESCENDING
                else:
                    # Peaked at/below release height, never approached the
                    # rim: dribble noise.
                    self._discard()
                    return None
            else:
                return None

        if self._phase is not ShotPhase.DESCENDING:
            return None

        if rim_box is None:
            # No rim geometry yet (early session, flaky rim detector): the
            # noise discard must still work or the machine wedges here.
            if (
                not ball.interpolated
                and not shot.entered_neighborhood
                and ball.y >= shot.release_y
            ):
                self._discard()
            return None

        if shot.crossed:
            return self._check_after_crossing(ball, rim_box, frame_index, px_per_m)
        return self._check_no_crossing(ball, rim_box, frame_index, px_per_m)

    # ------------------------------------------------------------- crossing

    def _detect_crossing(
        self,
        prev: Optional[metrics.Sample],
        ball: BallTrack,
        rim_box: Box,
        frame_index: int,
    ) -> None:
        """Register a downward rim-top crossing within the horizontal span.

        Runs in RISING and DESCENDING alike (positional descent decides, not
        the smoothed velocity), and across detection gaps — a passage that
        happened entirely inside a gap registers on the reappear frame,
        flagged occluded.
        """
        shot = self._shot
        if prev is None or ball.y <= prev[3]:
            return
        prev_frame, _, prev_x, prev_y, prev_interp = prev
        rim_x1, rim_top, rim_x2, _ = rim_box
        if not (prev_y < rim_top <= ball.y):
            return
        frac = (rim_top - prev_y) / (ball.y - prev_y)
        x_cross = prev_x + (ball.x - prev_x) * frac
        if not (rim_x1 <= x_cross <= rim_x2):
            return
        if shot.crossed:
            shot.rattled = True  # re-crossing after a bounce back above the rim
        else:
            shot.crossed = True
            shot.crossing_frame = frame_index
            shot.crossing_x = x_cross
            shot.crossing_occluded = (
                prev_interp or ball.interpolated or frame_index - prev_frame > 1
            )

    def _check_after_crossing(
        self, ball: BallTrack, rim_box: Box, frame_index: int, px_per_m: Optional[float]
    ) -> Optional[ShotEvent]:
        shot = self._shot
        rim_x1, rim_top, rim_x2, rim_bottom = rim_box

        if ball.interpolated:
            shot.post_cross_unobserved += 1
            return None

        tol = self._cfg.span_tolerance_frac * (rim_x2 - rim_x1)
        near_span = (rim_x1 - tol) <= ball.x <= (rim_x2 + tol)

        if ball.y > rim_bottom:
            if near_span:
                # Observed below the rim: make. (The crossing was strict
                # in-span; the reappearance check tolerates lateral net drift.)
                occluded = (
                    shot.crossing_occluded
                    or shot.post_cross_unobserved >= self._cfg.occlusion_frames_for_make
                )
                if shot.rattled:
                    conf = VerdictConfidence.RATTLED
                elif occluded:
                    conf = VerdictConfidence.OCCLUDED
                else:
                    conf = VerdictConfidence.CLEAN
                return self._resolve(Verdict.MAKE, conf, frame_index, px_per_m)
            # Below rim height but well wide of the span: bounced off and fell
            # beside the rim without ever being seen through the cylinder.
            return self._resolve(Verdict.MISS, VerdictConfidence.RATTLED, frame_index, px_per_m)

        # Still at/above the rim bottom. A pop-out is NOT resolved at the
        # first upward frame — the ball is still ballistic and may drop back
        # through (rattle-in). One rattle, one event: wait for either a
        # re-cross -> below (make) or a neighborhood exit (miss).
        if ball.vy < 0:
            shot.rattled = True
        if not self._in_neighborhood(ball, rim_box):
            conf = (
                VerdictConfidence.OCCLUDED
                if shot.post_cross_unobserved >= self._cfg.occlusion_frames_for_make
                else VerdictConfidence.RATTLED
            )
            return self._resolve(Verdict.MISS, conf, frame_index, px_per_m)
        return None

    def _check_no_crossing(
        self, ball: BallTrack, rim_box: Box, frame_index: int, px_per_m: Optional[float]
    ) -> Optional[ShotEvent]:
        shot = self._shot
        if ball.interpolated:
            return None  # never resolve on evidence nobody observed
        if shot.entered_neighborhood and not self._in_neighborhood(ball, rim_box):
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
