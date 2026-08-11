"""M4 renderer: (frame, FrameState) -> composed split-screen canvas, plus the
video sink that writes h264/yuv420p and muxes the source audio back in.

No shot logic lives here — if a verdict or a metric isn't already on the
FrameState, the fix goes in the engine, not in drawing code. The HUD only
*holds* display values between frames (the reference UI keeps the last shot's
numbers on screen).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from engine.config import EngineConfig
from engine.types import FrameState, PoseState
from render import draw
from render import layout as L
from render.text import TextDrawer


@dataclass
class HudState:
    """Display values persisted across frames (last shot stays on screen)."""

    shot_speed_kmh: Optional[float] = None
    current_speed_kmh: Optional[float] = None
    ball_px: Optional[float] = None
    distance_m: Optional[float] = None
    offset_x_m: Optional[float] = None
    offset_y_m: Optional[float] = None
    verdict: Optional[str] = None
    verdict_confidence: Optional[str] = None
    cross_x_m: Optional[float] = None
    shot_active: bool = False
    release_pose: Optional[PoseState] = None
    _active_shot: Optional[int] = None
    _flight_speeds: list[float] = field(default_factory=list)

    def update(self, state: FrameState) -> None:
        if state.current_speed_ms is not None:
            self.current_speed_kmh = state.current_speed_ms * 3.6
        if state.ball is not None and state.ball.bbox is not None:
            self.ball_px = state.ball.bbox[2] - state.ball.bbox[0]
        if state.distance_to_rim_m is not None:
            self.distance_m = state.distance_to_rim_m
        if (
            state.ball is not None
            and state.rim is not None
            and state.scale_px_per_m
        ):
            rim_cx = (state.rim.bbox[0] + state.rim.bbox[2]) / 2
            rim_cy = (state.rim.bbox[1] + state.rim.bbox[3]) / 2
            self.offset_x_m = (state.ball.x - rim_cx) / state.scale_px_per_m
            self.offset_y_m = (rim_cy - state.ball.y) / state.scale_px_per_m

        if state.active_shot_id is not None and state.active_shot_id != self._active_shot:
            # New attempt armed: clear the previous verdict, freeze the pose.
            self._active_shot = state.active_shot_id
            self.shot_active = True
            self.verdict = None
            self.verdict_confidence = None
            self.cross_x_m = None
            self.release_pose = state.pose
            self._flight_speeds = []
            self.shot_speed_kmh = None
        if self.shot_active and self.verdict is None and state.current_speed_ms:
            if len(self._flight_speeds) < 5:
                self._flight_speeds.append(state.current_speed_ms)
                self.shot_speed_kmh = (
                    sum(self._flight_speeds) / len(self._flight_speeds) * 3.6
                )

        for event in state.events:
            self.verdict = event.verdict.value
            self.verdict_confidence = event.verdict_confidence.value
            self.shot_active = False
            if event.release_velocity_ms is not None:
                self.shot_speed_kmh = event.release_velocity_ms * 3.6
            self.cross_x_m = self.offset_x_m
        if state.active_shot_id is None and self.verdict is None:
            self.shot_active = False


class Renderer:
    def __init__(
        self,
        config: Optional[EngineConfig] = None,
        initials: str = "AA",
        total_frames: Optional[int] = None,
    ):
        self.config = config or EngineConfig.upload()
        self.initials = initials
        self.total_frames = total_frames
        self.hud = HudState()
        self._text = TextDrawer()
        self._min_conf = self.config.pose.keypoint_confidence

    def draw(self, frame: np.ndarray, state: FrameState) -> np.ndarray:
        self.hud.update(state)
        canvas = np.zeros((L.CANVAS_H, L.CANVAS_W, 3), dtype=np.uint8)
        draw.draw_panel_chrome(canvas)

        pane, s, ox, oy = draw.fit_video(frame)
        draw.draw_video_overlays(
            pane, state, s, ox, oy, self._text,
            self.total_frames, self.initials, self._min_conf,
        )
        vx, vy, vw, vh = L.VIDEO_RECT
        canvas[vy : vy + vh, vx : vx + vw] = pane

        draw.draw_panels(canvas, self.hud, self._text)
        draw.draw_pose_inset(canvas, self.hud.release_pose, self._text, self._min_conf)
        draw.draw_joint_table(canvas, self.hud.release_pose, self._text, self._min_conf)
        return self._text.flush(canvas)


class VideoSink:
    """Writes composed canvases, then re-encodes to h264/yuv420p at source
    fps and muxes the source clip's audio back in (plan section 8)."""

    def __init__(self, out_path: str | Path, fps: float):
        import cv2

        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.out_path.with_suffix(".raw.mp4")
        self._writer = cv2.VideoWriter(
            str(self._tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (L.CANVAS_W, L.CANVAS_H)
        )

    def write(self, canvas: np.ndarray) -> None:
        self._writer.write(canvas)

    def close(self, audio_source: Optional[str | Path] = None) -> Path:
        self._writer.release()
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            self._tmp.replace(self.out_path)  # no ffmpeg: raw mp4v output
            return self.out_path
        base = [ffmpeg, "-y", "-i", str(self._tmp)]
        encode = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                  "-pix_fmt", "yuv420p"]
        if audio_source is not None:
            cmd = base + ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?",
                          "-c:a", "aac", "-shortest"] + encode + [str(self.out_path)]
        else:
            cmd = base + encode + [str(self.out_path)]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0 and audio_source is not None:
            # Source may have no audio stream in a mappable form: video-only.
            subprocess.run(base + encode + [str(self.out_path)], capture_output=True)
        self._tmp.unlink(missing_ok=True)
        return self.out_path
