"""ShotEngine: one frame plus timestamp in, one FrameState out.

The engine does not know where frames come from and performs no rendering and
no file I/O — sources feed it, the renderer and event log consume its output.
The detector and pose model are injected callables so tests can script
detections and production can wire RF-DETR without the engine changing.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Callable, Optional

from engine import form
from engine.calibration import ScaleCalibration
from engine.config import EngineConfig
from engine.state_machine import ShotStateMachine
from engine.tracker import BallTracker
from engine.types import Detection, FrameState, PoseState, ShotEvent, ShotPhase

Detector = Callable[[object], list[Detection]]
PoseModel = Callable[[object], Optional[PoseState]]


class ShotEngine:
    def __init__(
        self,
        config: EngineConfig,
        detector: Optional[Detector] = None,
        pose_model: Optional[PoseModel] = None,
    ):
        self.config = config
        self._detector = detector
        self._pose_model = pose_model
        self._tracker = BallTracker(config)
        self._machine = ShotStateMachine(config)
        self._calibration = ScaleCalibration(
            rim_diameter_m=config.rim_diameter_m,
            max_samples=config.calibration_frames,
            min_samples=config.calibration_min_samples,
            drift_frames=config.calibration_drift_frames,
        )
        self._frame_index = -1
        self._trail: deque[tuple[float, float]] = deque(maxlen=config.trail_length)
        self._last_t = 0.0
        self._dt_estimate = 1.0 / 30.0
        self._finalized = False
        self._last_pose: Optional[PoseState] = None
        # Recent (frame_index, pose, ball_xy) for release-frame form metrics:
        # a shot arms release_consecutive_frames after its release frame, so a
        # short ring suffices.
        self._recent: deque[tuple[int, Optional[PoseState], Optional[tuple[float, float]]]] = (
            deque(maxlen=8)
        )
        self._release_stash: dict[int, tuple[Optional[PoseState], Optional[tuple[float, float]]]] = {}
        self._stashed_shot: Optional[int] = None

    @property
    def calibration(self) -> ScaleCalibration:
        return self._calibration

    def process_frame(self, frame: object, t: float) -> FrameState:
        if self._finalized:
            raise RuntimeError(
                "process_frame after finalize(): the engine's frame clock has "
                "advanced past end-of-stream; use a fresh ShotEngine per source"
            )
        self._frame_index += 1
        if t > self._last_t:
            self._dt_estimate = t - self._last_t
        self._last_t = t

        detections = self._detector(frame) if self._detector else []
        ball_dets = [
            d
            for d in detections
            if d.class_name == "ball" and d.confidence >= self.config.detector.ball_confidence
        ]
        rim_dets = [
            d
            for d in detections
            if d.class_name == "rim" and d.confidence >= self.config.detector.rim_confidence
        ]

        rim = max(rim_dets, key=lambda d: d.confidence) if rim_dets else None
        if rim is not None:
            self._calibration.observe_rim(rim)

        # Rim geometry for shot logic: the session's median box once
        # calibration has one — the rim doesn't move within a session, and a
        # stable box keeps detection jitter out of the crossing checks. Live
        # detections only bridge the pre-calibration frames.
        median_box = self._calibration.median_rim_box
        rim_box = median_box if median_box is not None else (rim.bbox if rim is not None else None)

        ball = self._tracker.update(ball_dets, t)
        if ball is not None:
            self._trail.append((ball.x, ball.y))
        elif self._trail:
            self._trail.clear()

        pose = self._update_pose(frame)
        wrists = form.confident_wrists(pose, self.config.pose.keypoint_confidence)

        ball_xy = (ball.x, ball.y) if ball is not None and not ball.interpolated else None
        self._recent.append((self._frame_index, pose, ball_xy))

        px_per_m = self._calibration.px_per_m
        event = self._machine.update(
            ball, rim_box, self._frame_index, t, px_per_m,
            px_per_m_hint=self._calibration.px_per_m_hint,
            wrists=wrists,
        )
        self._stash_release_context()
        if event is not None:
            self._fill_form_metrics(event, px_per_m)

        return FrameState(
            frame_index=self._frame_index,
            t=t,
            phase=self._machine.phase,
            ball=ball,
            rim=rim,
            pose=pose,
            trail=list(self._trail),
            events=[event] if event else [],
            active_shot_id=self._machine.active_shot_id,
            scale_px_per_m=px_per_m,
            distance_to_rim_m=self._distance_to_rim_m(ball, rim_box, px_per_m),
            current_speed_ms=self._speed_ms(ball, px_per_m),
        )

    def _update_pose(self, frame: object) -> Optional[PoseState]:
        if self._pose_model is None or not self.config.pose.enabled:
            return None
        n = max(1, self.config.pose.every_n_frames)
        if self._frame_index % n == 0:
            self._last_pose = self._pose_model(frame)
        return self._last_pose  # carried forward on skipped frames (M5 cadence)

    def _stash_release_context(self) -> None:
        """When a shot arms, remember the pose + ball position at its release
        frame so form metrics can be computed at resolution time."""
        shot_id = self._machine.active_shot_id
        if shot_id is None or shot_id == self._stashed_shot:
            return
        self._stashed_shot = shot_id
        release_frame = self._machine.active_release_frame
        best = None
        for f, pose, ball_xy in self._recent:
            if abs(f - release_frame) <= 2 and (best is None or abs(f - release_frame) < best[0]):
                best = (abs(f - release_frame), pose, ball_xy)
        self._release_stash[shot_id] = (best[1], best[2]) if best else (None, None)
        while len(self._release_stash) > 4:
            self._release_stash.pop(next(iter(self._release_stash)))

    def _fill_form_metrics(self, event: ShotEvent, px_per_m: Optional[float]) -> None:
        pose, ball_xy = self._release_stash.pop(event.shot_id, (None, None))
        metrics = form.release_form_metrics(
            pose, ball_xy, px_per_m, self.config.pose.keypoint_confidence
        )
        event.elbow_deg = metrics["elbow_deg"]
        event.knee_deg = metrics["knee_deg"]
        event.shoulder_hip_deg = metrics["shoulder_hip_deg"]
        event.release_height_m = metrics["release_height_m"]

    def finalize(self) -> list[ShotEvent]:
        """The stream ended. A shot still open at end-of-stream is a real
        trajectory end (plan 6.2): drive the machine with no-ball frames until
        it resolves or discards, and return any events emitted. Clip-per-shot
        footage routinely cuts right at the outcome, so without this flush
        those attempts would silently vanish.
        """
        self._finalized = True
        events: list[ShotEvent] = []
        rim_box = self._calibration.median_rim_box
        px_per_m = self._calibration.px_per_m
        limit = self.config.occlusion_hold_frames + self.config.max_gap_frames + 3
        for _ in range(limit):
            if self._machine.phase is ShotPhase.IDLE:
                break
            self._frame_index += 1
            self._last_t += self._dt_estimate
            event = self._machine.update(
                None, rim_box, self._frame_index, self._last_t, px_per_m
            )
            if event:
                self._fill_form_metrics(event, px_per_m)
                events.append(event)
        return events

    @staticmethod
    def _distance_to_rim_m(ball, rim_box, px_per_m) -> Optional[float]:
        if ball is None or rim_box is None or px_per_m is None:
            return None
        rim_cx = (rim_box[0] + rim_box[2]) / 2
        rim_cy = (rim_box[1] + rim_box[3]) / 2
        return math.hypot(ball.x - rim_cx, ball.y - rim_cy) / px_per_m

    @staticmethod
    def _speed_ms(ball, px_per_m) -> Optional[float]:
        if ball is None or px_per_m is None or not ball.velocity_valid:
            return None  # a freshly seeded track has no measured speed yet
        return math.hypot(ball.vx, ball.vy) / px_per_m
