"""Drawing routines for the M4 layout: video-pane overlays, metric panels,
pose inset, joint table. Pure functions of (canvas, state/hud); no engine
logic — anything a panel needs must already be on FrameState or the HUD.
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from engine.types import FrameState, PoseState
from render import layout as L
from render.text import TextDrawer

SKELETON = (
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)

TABLE_ROWS = (
    ("shoulder-to-hip", "shoulder", "hip"),
    ("hip-to-knee", "hip", "knee"),
    ("knee-to-ankle", "knee", "ankle"),
)


def fit_video(frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """Scale the source frame into the video pane; returns (pane, s, ox, oy)."""
    _, _, pw, ph = L.VIDEO_RECT
    h, w = frame.shape[:2]
    s = min(pw / w, ph / h)
    nw, nh = int(w * s), int(h * s)
    ox, oy = (pw - nw) // 2, (ph - nh) // 2
    pane = np.full((ph, pw, 3), 16, dtype=np.uint8)
    pane[oy : oy + nh, ox : ox + nw] = cv2.resize(frame, (nw, nh))
    return pane, s, ox, oy


def draw_video_overlays(
    pane: np.ndarray,
    state: FrameState,
    s: float,
    ox: int,
    oy: int,
    text: TextDrawer,
    total_frames: Optional[int],
    initials: str,
    min_conf: float,
) -> None:
    def pt(x: float, y: float) -> tuple[int, int]:
        return int(x * s) + ox, int(y * s) + oy

    if state.rim is not None:
        x1, y1, x2, y2 = state.rim.bbox
        cv2.rectangle(pane, pt(x1, y1), pt(x2, y2), L.RIM_GREEN, 2)
        text.queue("rim", (pt(x1, y1)[0], pt(x1, y1)[1] - 22), 18, L.RIM_GREEN)

    if state.pose is not None:
        kp = state.pose.keypoints
        for a, b in SKELETON:
            pa, pb = kp.get(a), kp.get(b)
            if pa and pb and pa[2] >= min_conf and pb[2] >= min_conf:
                cv2.line(pane, pt(pa[0], pa[1]), pt(pb[0], pb[1]), L.PINK, 2)
        for name, (x, y, c) in kp.items():
            if c >= min_conf and "eye" not in name and "ear" not in name and name != "nose":
                cv2.circle(pane, pt(x, y), 3, L.PINK, -1)
        # Shooter tag with a red arrow above the head.
        tops = [
            kp[n] for n in ("nose", "left_shoulder", "right_shoulder")
            if kp.get(n) and kp[n][2] >= min_conf
        ]
        if tops:
            hx, hy = pt(
                sum(p[0] for p in tops) / len(tops), min(p[1] for p in tops)
            )
            tri = np.array(
                [[hx - 9, hy - 26], [hx + 9, hy - 26], [hx, hy - 12]], np.int32
            )
            cv2.fillPoly(pane, [tri], (40, 40, 230))
            (tw, th) = (13 * len(initials) + 10, 24)
            cv2.rectangle(
                pane, (hx - tw // 2, hy - 28 - th), (hx + tw // 2, hy - 28), L.BLACK, -1
            )
            text.queue(
                initials, (hx, hy - 28 - th // 2), 18, L.WHITE, stroke=1, anchor="mm"
            )

    for a, b in zip(state.trail, state.trail[1:]):
        cv2.line(pane, pt(*a), pt(*b), L.TRAIL_ORANGE, 3)
    if state.ball is not None:
        color = (150, 150, 150) if state.ball.interpolated else L.BALL_BLUE
        if state.ball.bbox is not None:
            x1, y1, x2, y2 = state.ball.bbox
            cv2.rectangle(pane, pt(x1, y1), pt(x2, y2), color, 2)
            text.queue("ball", (pt(x1, y1)[0], pt(x1, y1)[1] - 22), 18, L.BALL_BLUE)
        else:
            cv2.circle(pane, pt(state.ball.x, state.ball.y), 10, color, 2)

    counter = f"FRAME {state.frame_index}"
    if total_frames:
        counter += f" / {total_frames - 1}"
    _, _, pw, ph = L.VIDEO_RECT
    text.queue(counter, (pw - 24, ph - 18), L.COUNTER_SIZE, L.WHITE, anchor="rs")


def draw_panel_chrome(canvas: np.ndarray) -> None:
    canvas[:, L.PANEL_X :] = L.CREAM
    strip_y = L.VIDEO_RECT[3]
    canvas[strip_y:, : L.PANEL_X] = L.CREAM
    for i in (1, 2):
        y = i * L.PANEL_H
        cv2.line(canvas, (L.PANEL_X, y), (L.CANVAS_W, y), L.LINE, 3)
    cv2.line(canvas, (L.PANEL_X, 0), (L.PANEL_X, L.CANVAS_H), L.LINE, 3)
    cv2.line(canvas, (0, strip_y), (L.PANEL_X, strip_y), L.LINE, 3)
    for x, y, w, h in (L.INSET_RECT, L.TABLE_RECT):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), L.LINE, 2)


def draw_panels(canvas: np.ndarray, hud, text: TextDrawer) -> None:
    x = L.PANEL_X + L.PANEL_PAD

    # Panel 1: SHOT SPEED
    y = 0
    text.queue("SHOT SPEED", (x, y + 40), L.HEADER_SIZE, L.ORANGE, stroke=1)
    big = f"{hud.shot_speed_kmh:.0f} KM/H" if hud.shot_speed_kmh is not None else "--"
    text.queue(big, (x, y + 116), L.BIG_SIZE, L.INK, stroke=2)
    cur = f"{hud.current_speed_kmh:.0f}" if hud.current_speed_kmh is not None else "--"
    text.queue(f"CURRENT SPEED: {cur} KM/H", (x, y + 268), L.SMALL_SIZE, L.GRAY)
    bpx = f"{hud.ball_px:.1f}" if hud.ball_px is not None else "--"
    text.queue(f"BALL PX: {bpx}", (x, y + 306), L.SMALL_SIZE, L.GRAY)

    # Panel 2: BALL DISTANCE TO RIM
    y = L.PANEL_H
    text.queue("BALL DISTANCE TO", (x, y + 40), L.HEADER_SIZE, L.ORANGE, stroke=1)
    text.queue("RIM", (x, y + 90), L.HEADER_SIZE, L.ORANGE, stroke=1)
    big = f"{hud.distance_m:.1f} M" if hud.distance_m is not None else "-.- M"
    text.queue(big, (x, y + 162), L.BIG_SIZE, L.INK, stroke=2)
    if hud.offset_x_m is not None:
        off = f"X: {hud.offset_x_m:+.2f}M  Y: {hud.offset_y_m:+.2f}M"
    else:
        off = "X: --  Y: --"
    text.queue(off, (x, y + 312), L.SMALL_SIZE, L.GRAY)

    # Panel 3: MAKE?
    y = 2 * L.PANEL_H
    text.queue("MAKE?", (x, y + 40), L.HEADER_SIZE, L.ORANGE, stroke=1)
    if hud.verdict is None:
        text.queue("NO", (x, y + 116), L.BIG_SIZE, L.SILVER, stroke=2)
        status = (
            "CROSS DETECTOR: ACTIVE" if hud.shot_active else "CROSS DETECTOR: WAITING"
        )
        text.queue(status, (x, y + 276), L.SMALL_SIZE, L.GRAY)
    else:
        made = hud.verdict == "make"
        text.queue(
            "MAKE!!" if made else "MISS",
            (x, y + 116), L.BIG_SIZE, L.GREEN if made else L.RED, stroke=2,
        )
        if hud.cross_x_m is not None:
            text.queue(
                f"CROSS AT X: {hud.cross_x_m:+.2f}M",
                (x, y + 276), L.SMALL_SIZE, L.GRAY,
            )
        text.queue(f"({hud.verdict_confidence})", (x, y + 314), L.SMALL_SIZE, L.GRAY)


def draw_pose_inset(canvas: np.ndarray, pose: Optional[PoseState], text: TextDrawer, min_conf: float) -> None:
    x, y, w, h = L.INSET_RECT
    text.queue("Pose at Release", (x + 12, y + 34), 24, L.ORANGE, stroke=1)
    if pose is None:
        text.queue("(no pose)", (x + 12, y + 80), L.SMALL_SIZE, L.SILVER)
        return
    pts = {
        n: (px, py)
        for n, (px, py, c) in pose.keypoints.items()
        if c >= min_conf
    }
    if len(pts) < 4:
        text.queue("(low confidence)", (x + 12, y + 80), L.SMALL_SIZE, L.SILVER)
        return
    xs, ys = [p[0] for p in pts.values()], [p[1] for p in pts.values()]
    bw, bh = max(xs) - min(xs) or 1.0, max(ys) - min(ys) or 1.0
    inner_x, inner_y, inner_w, inner_h = x + 30, y + 60, w - 60, h - 90
    s = min(inner_w / bw, inner_h / bh)
    ox = inner_x + (inner_w - bw * s) / 2 - min(xs) * s
    oy = inner_y + (inner_h - bh * s) / 2 - min(ys) * s

    def pt(n: str) -> tuple[int, int]:
        px, py = pts[n]
        return int(px * s + ox), int(py * s + oy)

    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(canvas, pt(a), pt(b), L.PINK, 3)
    for n in pts:
        cv2.circle(canvas, pt(n), 5, L.PINK, -1)


def segment_angle_signed(a: tuple, b: tuple) -> float:
    """Angle of the (upper joint -> lower joint) segment vs vertical, signed
    by horizontal direction; the reference table's convention."""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))


def draw_joint_table(canvas: np.ndarray, pose: Optional[PoseState], text: TextDrawer, min_conf: float) -> None:
    x, y, w, h = L.TABLE_RECT
    col1, col2, col3 = x + 16, x + w - 340, x + w - 170
    text.queue("Joint Angles", (col1, y + 24), 24, L.ORANGE, stroke=1)
    text.queue("Left Side", (col2, y + 24), L.TABLE_SIZE, L.ORANGE)
    text.queue("Right Side", (col3, y + 24), L.TABLE_SIZE, L.ORANGE)
    cv2.line(canvas, (x, y + 66), (x + w, y + 66), L.LINE, 1)

    def joint(side: str, name: str):
        kp = pose.keypoints.get(f"{side}_{name}") if pose else None
        return (kp[0], kp[1]) if kp and kp[2] >= min_conf else None

    for i, (label, upper, lower) in enumerate(TABLE_ROWS):
        ry = y + 110 + i * 90
        text.queue(label, (col1, ry), L.TABLE_SIZE, L.INK)
        for col, side in ((col2, "left"), (col3, "right")):
            a, b = joint(side, upper), joint(side, lower)
            val = f"{segment_angle_signed(a, b):.0f}°" if a and b else "--"
            text.queue(val, (col, ry), L.TABLE_SIZE, L.INK)
        if i < len(TABLE_ROWS) - 1:
            cv2.line(canvas, (x, ry + 28), (x + w, ry + 28), (200, 205, 210), 1)
