"""Shot state machine (plan 6.2, 6.3).

IDLE -> RISING -> DESCENDING -> RESOLVED(make|miss) -> IDLE

- IDLE -> RISING: release detected — sustained upward velocity over
  ``release_consecutive_frames`` observed frames with measured velocity. After
  a resolution, no new release can arm while the ball is inside the rim
  neighborhood until it has been observed outside once (residual rim-bounce
  flight is not a release; cold-start close-range releases are unaffected).
  M3 adds the wrist-separation half of the rule.
- RISING -> DESCENDING: vertical velocity sign flip while the ball is above
  release height (or already inside the rim neighborhood — a bank shot's
  rebound off glass must not reset the machine), or a positional rim-top
  crossing observed before the smoothed velocity has flipped (flat arcs at
  rim height live in that 1-2 frame EMA lag window).
- DESCENDING -> RESOLVED: make/miss rules below, ball exits the rim
  neighborhood, the trajectory ends after an ``occlusion_hold_frames``
  reappear window (a net occlusion routinely outlives the tracker's 4-frame
  extrapolation, so track loss alone must not resolve the shot), or the
  ``max_shot_seconds`` hard cap lands (nothing may wedge the machine).
- Noise discards (no event): a candidate that never rises toward the rim
  neighborhood (plan 6.2 dribble rule), and any uncrossed candidate whose
  apex never reached the rim band (min observed y above the rim bottom) —
  floor bounces under the hoop must not log phantom misses. Revisit both
  gates at M2 with real footage.

Make (6.3): centroid crosses the rim's top edge moving downward within the
rim's horizontal span — on an *observed, consecutive-frame* segment; a
crossing can never be synthesized from tracker ghosts or across detection
gaps, which is what keeps behind-the-rim airballs from faking makes (plan
12) — then is observed below the rim near the span (``span_tolerance_frac``
absorbs lateral net drift). Track loss after an observed crossing holds the
shot open; a reappearance below the rim within the hold window is the
occluded make.

Rattles (plan 12 calls them the M2 error budget; M6 rim keypoints are the
real fix): a pop-out above the rim is not resolved at the first upward frame.
If the pop stays below ``rattle_pop_max_rim_widths`` above the rim top, the
ball may drop back through — an observed re-cross followed by a below-rim
observation scores a rattled make; leaving the neighborhood scores a rattled
miss. A pop higher than that is a rejection and resolves a rattled miss
immediately. In mono 2D a small-hop front-rim out that falls in-span is
indistinguishable from a rattle-in; this policy takes the physically likelier
reading and always stamps confidence=rattled so M2 can audit the population.

``verdict_confidence`` (clean | rattled | occluded) is logged for auditing,
never silently guessed: resolutions reached right after unobserved stretches
are stamped occluded, not clean.

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
    min_observed_y: Optional[float] = None  # apex of the observed trajectory
    crossed: bool = False
    crossing_frame: Optional[int] = None
    crossing_x: Optional[float] = None
    crossing_occluded: bool = False  # the crossing hid inside a detection gap
    # A below-cylinder reappearance awaiting its fall-speed verdict:
    # (reappear_frame, reappear_y, ballistic_fall_px_per_frame, x_cross)
    pending_bridge: Optional[tuple[int, float, float, float]] = None
    last_observed: Optional[metrics.Sample] = None
    popped_out: bool = False  # observed above the rim top after crossing
    recrossed_after_pop: bool = False
    post_cross_unobserved: int = 0  # unobserved frames since the crossing
    unseen_streak: int = 0  # consecutive frames with no track at all
    last_gap: int = 0  # unobserved run immediately before the current observation
    gap_near_rim: bool = False  # a long unobserved stretch happened near the rim
    _nonobserved_run: int = 0
    rattled: bool = False


class ShotStateMachine:
    def __init__(self, config: EngineConfig):
        self._cfg = config
        self._phase = ShotPhase.IDLE
        self._next_shot_id = 1
        self._streak: list[metrics.Sample] = []  # consecutive rising observed samples
        self._shot: Optional[_ActiveShot] = None
        self._await_exit = False  # post-resolution: ball must leave the rim area once
        self._last_rim_box: Optional[Box] = None

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
        px_per_m_hint: Optional[float] = None,
    ) -> Optional[ShotEvent]:
        """Advance one frame; returns a ShotEvent on the frame a verdict lands.

        ``px_per_m`` feeds metrics (strict: None -> null). ``px_per_m_hint``
        feeds thresholds only and may be a last-known scale surviving a
        calibration reset."""
        if rim_box is not None:
            self._last_rim_box = rim_box
        else:
            # The rim doesn't move within a session: geometry seen earlier
            # keeps working through detector dropouts and calibration resets.
            rim_box = self._last_rim_box

        if self._phase is ShotPhase.RESOLVED:
            self._phase = ShotPhase.IDLE

        if self._phase is ShotPhase.IDLE:
            self._update_idle(ball, rim_box, frame_index, t, px_per_m_hint or px_per_m)
            return None
        return self._update_airborne(ball, rim_box, frame_index, t, px_per_m)

    # ------------------------------------------------------------------ IDLE

    def _update_idle(
        self,
        ball: Optional[BallTrack],
        rim_box: Optional[Box],
        frame_index: int,
        t: float,
        scale_hint: Optional[float],
    ) -> None:
        observed = ball is not None and not ball.interpolated
        if self._await_exit and observed:
            # The previous shot's ball must be seen leaving the rim area
            # *downward* (outside the neighborhood, below rim height) before a
            # new release can arm — a rebound climbing above the neighborhood
            # is still the old shot's flight, not a release.
            if rim_box is None:
                self._await_exit = False
            elif not self._in_neighborhood(ball, rim_box) and ball.y > (
                (rim_box[1] + rim_box[3]) / 2
            ):
                self._await_exit = False

        # Threshold in m/s once the session has (or ever had) a scale — a
        # launch is ~7 m/s, raising the ball into the pocket ~1-2 m/s; the px
        # fallback covers only the never-calibrated cold start.
        if scale_hint is not None:
            min_up = self._cfg.release_min_upward_speed_ms * scale_hint
        else:
            min_up = self._cfg.release_min_upward_speed_px_s
        rising = (
            observed
            and not self._await_exit
            and ball.velocity_valid
            and ball.vy <= -min_up
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
            min_observed_y=min(s[3] for s in self._streak),
            last_observed=self._streak[-1],
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
        # The hard cap is checked AFTER this frame's evidence: a make observed
        # on the very frame the clock lands must win over the timeout, and the
        # clock's anchor can sit seconds early when a windup armed the release.
        timed_out = t - shot.release_t > self._cfg.max_shot_seconds

        if ball is None:
            # Track lost. Hold the shot open for a reappear window — net/body
            # occlusion at the rim routinely outlives the tracker's gap limit.
            shot.unseen_streak += 1
            shot._nonobserved_run += 1
            if shot.crossed:
                shot.post_cross_unobserved += 1
            if shot.unseen_streak <= self._cfg.occlusion_hold_frames and not timed_out:
                return None
            return self._resolve_ended(rim_box, frame_index, px_per_m)
        shot.unseen_streak = 0

        prev_obs = shot.last_observed
        self._record(ball, rim_box, frame_index, t)
        if rim_box is not None:
            self._detect_crossing(prev_obs, ball, rim_box, frame_index)

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

        if self._phase is not ShotPhase.DESCENDING:
            if timed_out:
                return self._resolve_ended(rim_box, frame_index, px_per_m)
            return None

        if rim_box is None:
            # No rim geometry ever seen this session: the noise discard must
            # still work or the machine wedges here.
            if (
                not ball.interpolated
                and not shot.entered_neighborhood
                and ball.y >= shot.release_y
            ):
                self._discard()
            elif timed_out:
                return self._resolve_ended(rim_box, frame_index, px_per_m)
            return None

        if shot.crossed:
            event = self._check_after_crossing(ball, rim_box, frame_index, px_per_m)
        else:
            event = self._check_no_crossing(ball, rim_box, frame_index, px_per_m)
        if event is not None or self._shot is None:  # resolved or discarded above
            return event
        if timed_out:
            return self._resolve_ended(rim_box, frame_index, px_per_m)
        return None

    # ------------------------------------------------------------- crossing

    def _detect_crossing(
        self,
        prev_obs: Optional[metrics.Sample],
        ball: BallTrack,
        rim_box: Box,
        frame_index: int,
    ) -> None:
        """Register a downward rim-top crossing within the horizontal span.

        Both endpoints must be *observed* — tracker ghosts never cross. A
        consecutive-frame segment always qualifies; a segment bridging a short
        detection gap qualifies only when both endpoints hug the rim (last
        seen just above it, reappearing inside/just under the cylinder). The
        net occludes the ball exactly at the crossing on real 30 fps footage
        (both missed makes of session-2026-08-11), but a behind-the-rim
        airball that vanishes and reappears on the ground far below must
        still never fabricate a crossing (plan 12).
        """
        shot = self._shot
        if ball.interpolated:
            return
        self._settle_pending_bridge(ball, frame_index)
        if prev_obs is None:
            return
        po_frame, _, po_x, po_y, _ = prev_obs
        gap = frame_index - po_frame
        if gap < 1 or gap > self._cfg.crossing_bridge_max_frames:
            return
        if ball.y <= po_y:
            return
        rim_x1, rim_top, rim_x2, rim_bottom = rim_box
        if not (po_y < rim_top <= ball.y):
            return
        if gap > 1:
            # Bridged crossing: 2D evidence for a crossing that hid in the
            # gap. Windows denominate in rim WIDTH — the calibrated,
            # drift-guarded dimension — never the flappy box height.
            if shot.crossed:
                return  # re-crosses after a pop-out must be observed, never bridged
            rim_w = rim_x2 - rim_x1
            if not (rim_x1 <= ball.x <= rim_x2):
                return  # reappeared out of span: never through the cylinder
            if po_y < rim_top - 2 * rim_w:
                return  # last seen implausibly far above the rim
            if ball.y <= rim_bottom:
                # Reappeared INSIDE the cylinder: the strongest 2D evidence
                # (both real occluded makes of session-2026-08-11 did this).
                pass
            elif ball.y <= rim_bottom + rim_w:
                # The net-exit zone just below the cylinder — also where a
                # behind-the-rim airball reappears, and where the tracker may
                # adopt a resting second ball. The discriminator is the fall
                # speed on the NEXT observed frame vs ballistic continuation:
                # a net transit keeps falling but meaningfully slowed; free
                # flight falls at ~100% of ballistic; a resting ball at ~0%.
                # Park the candidate and let _settle_pending_bridge decide.
                ballistic = self._extrapolate_pre_gap(shot, frame_index)
                if ballistic is None or ballistic[2] <= 0:
                    return  # cannot judge -> conservative reject (audit: miss)
                frac = (rim_top - po_y) / (ball.y - po_y)
                x_cross = po_x + (ball.x - po_x) * frac
                if rim_x1 <= x_cross <= rim_x2:
                    shot.pending_bridge = (frame_index, ball.y, ballistic[2], x_cross)
                return
            else:
                return  # reappeared far below: ground/behind-rim territory
        frac = (rim_top - po_y) / (ball.y - po_y)
        x_cross = po_x + (ball.x - po_x) * frac
        if not (rim_x1 <= x_cross <= rim_x2):
            return
        if shot.crossed:
            shot.rattled = True  # re-crossing after a bounce back above the rim
            if shot.popped_out:
                shot.recrossed_after_pop = True
        else:
            shot.crossed = True
            shot.crossing_frame = frame_index
            shot.crossing_x = x_cross
            shot.crossing_occluded = gap > 1

    def _settle_pending_bridge(self, ball: BallTrack, frame_index: int) -> None:
        """Decide a parked below-cylinder bridge on the next observed frame:
        confirm the crossing when the measured fall speed is 10-90% of the
        ballistic continuation (net drag), reject free-fall (~100%, airball
        behind the rim) and near-rest (~0%, adopted second ball)."""
        shot = self._shot
        if shot.pending_bridge is None:
            return
        if shot.crossed:
            shot.pending_bridge = None
            return
        pf, py_, ballistic_fall, x_cross = shot.pending_bridge
        since = frame_index - pf
        if since < 1:
            return
        shot.pending_bridge = None  # one observed frame decides, either way
        if since > 3:
            return  # too stale to trust the fall-speed measurement
        measured = (ball.y - py_) / since
        if 0.1 * ballistic_fall <= measured <= 0.9 * ballistic_fall:
            shot.crossed = True
            shot.crossing_frame = pf
            shot.crossing_x = x_cross
            shot.crossing_occluded = True

    @staticmethod
    def _extrapolate_pre_gap(
        shot: _ActiveShot, frame_index: int
    ) -> Optional[tuple[float, float, float]]:
        """Quadratic (constant-acceleration) extrapolation of the pre-gap
        track to ``frame_index``, from the last three consecutively observed
        samples: returns (x, y, fall_px_per_frame_at_next_step). Pure
        ballistic flight extrapolates onto itself almost exactly; contact
        shows up as a fall-speed change. None when the pre-gap track is too
        broken to judge."""
        observed = [s for s in shot.samples[:-1] if not s[4]][-3:]
        if len(observed) < 3:
            return None
        (f1, _, x1, y1, _), (f2, _, x2, y2, _), (f3, _, x3, y3, _) = observed
        if f2 - f1 != 1 or f3 - f2 != 1:
            return None
        n = frame_index - f3
        tri = n * (n + 1) / 2
        d2x = (x3 - x2) - (x2 - x1)
        d2y = (y3 - y2) - (y2 - y1)
        x = x3 + n * (x3 - x2) + tri * d2x
        y = y3 + n * (y3 - y2) + tri * d2y
        fall_next = (y3 - y2) + (n + 1) * d2y
        return x, y, fall_next

    def _check_after_crossing(
        self, ball: BallTrack, rim_box: Box, frame_index: int, px_per_m: Optional[float]
    ) -> Optional[ShotEvent]:
        shot = self._shot
        rim_x1, rim_top, rim_x2, rim_bottom = rim_box

        if ball.interpolated:
            shot.post_cross_unobserved += 1
            return None

        rim_width = rim_x2 - rim_x1
        tol = self._cfg.span_tolerance_frac * rim_width
        near_span = (rim_x1 - tol) <= ball.x <= (rim_x2 + tol)
        occluded = (
            shot.crossing_occluded
            or shot.post_cross_unobserved >= self._cfg.occlusion_frames_for_make
        )

        if ball.y > rim_bottom:
            if near_span and not (shot.popped_out and not shot.recrossed_after_pop):
                # Observed below the rim: make. (The crossing was strict
                # in-span; the reappearance check tolerates lateral net
                # drift.) After a pop-out, only an observed re-cross puts the
                # ball back through the cylinder.
                if shot.rattled:
                    conf = VerdictConfidence.RATTLED
                elif occluded:
                    conf = VerdictConfidence.OCCLUDED
                else:
                    conf = VerdictConfidence.CLEAN
                return self._resolve(Verdict.MAKE, conf, frame_index, px_per_m)
            # Below rim height but wide of the span, or below after an
            # unrecrossed pop-out: it fell outside the cylinder.
            conf = VerdictConfidence.OCCLUDED if occluded else VerdictConfidence.RATTLED
            return self._resolve(Verdict.MISS, conf, frame_index, px_per_m)

        # Still at/above the rim bottom.
        if ball.y < rim_top:
            # Observed back above the rim: a pop-out. Not resolved at this
            # first frame — the ball is still ballistic and may drop back
            # through (rattle-in). One rattle, one event.
            shot.rattled = True
            shot.popped_out = True
            if rim_top - ball.y > self._cfg.rattle_pop_max_rim_widths * rim_width:
                # Popped far above the rim: that's a rejection, not a rattle.
                return self._resolve(
                    Verdict.MISS, VerdictConfidence.RATTLED, frame_index, px_per_m
                )
        if ball.vy < 0:
            shot.rattled = True
        if not self._in_neighborhood(ball, rim_box):
            conf = VerdictConfidence.OCCLUDED if occluded else VerdictConfidence.RATTLED
            return self._resolve(Verdict.MISS, conf, frame_index, px_per_m)
        return None

    def _check_no_crossing(
        self, ball: BallTrack, rim_box: Box, frame_index: int, px_per_m: Optional[float]
    ) -> Optional[ShotEvent]:
        shot = self._shot
        if ball.interpolated:
            return None  # never resolve on evidence nobody observed
        rim_top = rim_box[1]
        if (
            shot.entered_neighborhood
            and not self._in_neighborhood(ball, rim_box)
            and ball.y > rim_top
        ):
            # Left the rim area, at/below rim level, without ever crossing the
            # top edge. (A ball above the neighborhood's ceiling is still
            # inbound — steep floaters poke over it near a square-ish rim box —
            # so exits through the TOP never resolve; the ball either comes
            # back down through the rim or drops below rim level out wide.)
            if not self._reached_rim_band(shot, rim_box):
                self._discard()  # under-rim floor bounce, not an attempt
                return None
            conf = (
                VerdictConfidence.OCCLUDED
                if shot.gap_near_rim
                or shot.last_gap >= self._cfg.occlusion_frames_for_make
                else VerdictConfidence.CLEAN
            )
            return self._resolve(Verdict.MISS, conf, frame_index, px_per_m)
        if not shot.entered_neighborhood and ball.y >= shot.release_y:
            # Fell back to release height without approaching the rim.
            self._discard()
        return None

    def _resolve_ended(
        self, rim_box: Optional[Box], frame_index: int, px_per_m: Optional[float]
    ) -> Optional[ShotEvent]:
        """The trajectory is over (hold expired or shot timed out) without a
        verdict observation. Miss if this was a real attempt; discard if not."""
        shot = self._shot
        if not shot.entered_neighborhood:
            self._discard()
            return None
        if not shot.crossed and rim_box is not None and not self._reached_rim_band(shot, rim_box):
            self._discard()
            return None
        return self._resolve(Verdict.MISS, VerdictConfidence.OCCLUDED, frame_index, px_per_m)

    # ------------------------------------------------------------- plumbing

    def _record(
        self, ball: BallTrack, rim_box: Optional[Box], frame_index: int, t: float
    ) -> None:
        shot = self._shot
        shot.samples.append((frame_index, t, ball.x, ball.y, ball.interpolated))
        if ball.interpolated:
            shot._nonobserved_run += 1
        else:
            shot.last_gap = shot._nonobserved_run
            if shot.last_gap >= self._cfg.occlusion_frames_for_make and shot.entered_neighborhood:
                shot.gap_near_rim = True  # the audit flag must survive to resolution
            shot._nonobserved_run = 0
            shot.last_observed = shot.samples[-1]
            if shot.min_observed_y is None or ball.y < shot.min_observed_y:
                shot.min_observed_y = ball.y
        if rim_box is not None and self._in_neighborhood(ball, rim_box):
            shot.entered_neighborhood = True

    def _in_neighborhood(self, ball: BallTrack, rim_box: Box) -> bool:
        x1, y1, x2, y2 = rim_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half = self._cfg.rim_neighborhood_scale * max(x2 - x1, y2 - y1) / 2
        return abs(ball.x - cx) <= half and abs(ball.y - cy) <= half

    @staticmethod
    def _reached_rim_band(shot: _ActiveShot, rim_box: Box) -> bool:
        """Did the observed apex get at least up to the rim's bottom edge?"""
        return shot.min_observed_y is not None and shot.min_observed_y <= rim_box[3]

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
        self._await_exit = True
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
