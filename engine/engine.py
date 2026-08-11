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

    @property
    def calibration(self) -> ScaleCalibration:
        return self._calibration

    def process_frame(self, frame: object, t: float) -> FrameState:
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

        px_per_m = self._calibration.px_per_m
        event = self._machine.update(ball, rim_box, self._frame_index, t, px_per_m)

        pose = self._pose_model(frame) if self._pose_model and self.config.pose.enabled else None

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

    def finalize(self) -> list[ShotEvent]:
        """The stream ended. A shot still open at end-of-stream is a real
        trajectory end (plan 6.2): drive the machine with no-ball frames until
        it resolves or discards, and return any events emitted. Clip-per-shot
        footage routinely cuts right at the outcome, so without this flush
        those attempts would silently vanish.
        """
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
