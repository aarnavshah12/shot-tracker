"""Drawing routines for the M4 layout: video-pane overlays, metric panels,
pose inset, joint table. Pure functions of (canvas, state/hud); no engine
logic — anything a panel needs must already be on FrameState or the HUD.

Style: Roboflow brand. Detection boxes are drawn Roboflow-style — 2px, square
corners, filled class+confidence label tab flush with the top-left corner.
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


def detection_box(
    pane: np.ndarray,
    text: TextDrawer,
    p1: tuple[int, int],
    p2: tuple[int, int],
    label: str,
    color: tuple[int, int, int],
    confidence: Optional[float] = None,
) -> None:
    """Roboflow-style box: 2px square corners + filled label tab, top-left."""
    cv2.rectangle(pane, p1, p2, color, 2)
    tab = label if confidence is None else f"{label} {confidence:.2f}"
    tw = int(len(tab) * 9.5) + 12
    x1, y1 = p1
    cv2.rectangle(pane, (x1 - 1, y1 - 22), (x1 - 1 + tw, y1), color, -1)
    text.queue(tab, (x1 + 5, y1 - 11), 16, L.WHITE, weight=500, anchor="lm")


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
        detection_box(
            pane, text, pt(x1, y1), pt(x2, y2), "rim", L.RIM_COLOR,
            state.rim.confidence,
        )

    if state.pose is not None:
        kp = state.pose.keypoints
        for a, b in SKELETON:
            pa, pb = kp.get(a), kp.get(b)
            if pa and pb and pa[2] >= min_conf and pb[2] >= min_conf:
                cv2.line(pane, pt(pa[0], pa[1]), pt(pb[0], pb[1]), L.SKELETON_COLOR, 2)
        for name, (x, y, c) in kp.items():
            if c >= min_conf and "eye" not in name and "ear" not in name and name != "nose":
                cv2.circle(pane, pt(x, y), 3, L.SKELETON_COLOR, -1)
        # Shooter tag above the head, Violet 600 like a class label.
        tops = [
            kp[n] for n in ("nose", "left_shoulder", "right_shoulder")
            if kp.get(n) and kp[n][2] >= min_conf
        ]
        if tops:
            hx, hy = pt(
                sum(p[0] for p in tops) / len(tops), min(p[1] for p in tops)
            )
            tri = np.array(
                [[hx - 8, hy - 24], [hx + 8, hy - 24], [hx, hy - 11]], np.int32
            )
            cv2.fillPoly(pane, [tri], L.TAG_COLOR)
            tw, th = 12 * len(initials) + 14, 24
            cv2.rectangle(
                pane, (hx - tw // 2, hy - 26 - th), (hx + tw // 2, hy - 26),
                L.TAG_COLOR, -1,
            )
            text.queue(
                initials, (hx, hy - 26 - th // 2), 17, L.WHITE, weight=600, anchor="mm"
            )

    for a, b in zip(state.trail, state.trail[1:]):
        cv2.line(pane, pt(*a), pt(*b), L.TRAIL_COLOR, 3)
    if state.ball is not None:
        if state.ball.interpolated:
            cv2.circle(pane, pt(state.ball.x, state.ball.y), 10, L.SILVER, 2)
        elif state.ball.bbox is not None:
            x1, y1, x2, y2 = state.ball.bbox
            detection_box(
                pane, text, pt(x1, y1), pt(x2, y2), "ball", L.BALL_COLOR,
                state.ball.confidence,
            )
        else:
            cv2.circle(pane, pt(state.ball.x, state.ball.y), 10, L.BALL_COLOR, 2)

    counter = f"FRAME {state.frame_index}"
    if total_frames:
        counter += f" / {total_frames - 1}"
    _, _, pw, ph = L.VIDEO_RECT
    text.queue(counter, (pw - 24, ph - 18), L.COUNTER_SIZE, L.WHITE, weight=500, anchor="rs")


def draw_panel_chrome(canvas: np.ndarray) -> None:
    canvas[:, L.PANEL_X :] = L.SURFACE
    strip_y = L.VIDEO_RECT[3]
    canvas[strip_y:, : L.PANEL_X] = L.SURFACE
    for i in (1, 2):
        y = i * L.PANEL_H
        cv2.line(canvas, (L.PANEL_X, y), (L.CANVAS_W, y), L.LINE, 2)
    cv2.line(canvas, (L.PANEL_X, 0), (L.PANEL_X, L.CANVAS_H), L.LINE, 2)
    cv2.line(canvas, (0, strip_y), (L.PANEL_X, strip_y), L.LINE, 2)
    for x, y, w, h in (L.INSET_RECT, L.TABLE_RECT):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), L.SURFACE_ALT, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), L.LINE, 1)


def draw_panels(canvas: np.ndarray, hud, text: TextDrawer) -> None:
    x = L.PANEL_X + L.PANEL_PAD

    # Panel 1: SHOT SPEED
    y = 0
    text.queue("SHOT SPEED", (x, y + 36), L.HEADER_SIZE, L.VIOLET_TEXT, weight=700)
    big = f"{hud.shot_speed_kmh:.0f} KM/H" if hud.shot_speed_kmh is not None else "--"
    text.queue(big, (x, y + 100), L.BIG_SIZE, L.INK, weight=700)
    cur = f"{hud.current_speed_kmh:.0f}" if hud.current_speed_kmh is not None else "--"
    text.queue(f"CURRENT SPEED: {cur} KM/H", (x, y + 254), L.SMALL_SIZE, L.GRAY, weight=500)
    bpx = f"{hud.ball_px:.1f}" if hud.ball_px is not None else "--"
    text.queue(f"BALL PX: {bpx}", (x, y + 286), L.SMALL_SIZE, L.GRAY, weight=500)

    # Panel 2: BALL DISTANCE TO RIM
    y = L.PANEL_H
    text.queue("BALL DISTANCE TO", (x, y + 36), L.HEADER_SIZE, L.VIOLET_TEXT, weight=700)
    text.queue("RIM", (x, y + 74), L.HEADER_SIZE, L.VIOLET_TEXT, weight=700)
    big = f"{hud.distance_m:.1f} M" if hud.distance_m is not None else "-.- M"
    text.queue(big, (x, y + 138), L.BIG_SIZE, L.INK, weight=700)
    if hud.offset_x_m is not None:
        off = f"X: {hud.offset_x_m:+.2f}M  Y: {hud.offset_y_m:+.2f}M"
    else:
        off = "X: --  Y: --"
    text.queue(off, (x, y + 286), L.SMALL_SIZE, L.GRAY, weight=500)

    # Panel 3: MAKE?
    y = 2 * L.PANEL_H
    text.queue("MAKE?", (x, y + 36), L.HEADER_SIZE, L.VIOLET_TEXT, weight=700)
    if hud.verdict is None:
        text.queue("NO", (x, y + 100), L.BIG_SIZE, L.SILVER, weight=700)
        status = (
            "CROSS DETECTOR: ACTIVE" if hud.shot_active else "CROSS DETECTOR: WAITING"
        )
        text.queue(status, (x, y + 254), L.SMALL_SIZE, L.GRAY, weight=500)
    else:
        made = hud.verdict == "make"
        text.queue(
            "MAKE!!" if made else "MISS",
            (x, y + 100), L.BIG_SIZE, L.GREEN if made else L.RED, weight=700,
        )
        if hud.cross_x_m is not None:
            text.queue(
                f"CROSS AT X: {hud.cross_x_m:+.2f}M",
                (x, y + 254), L.SMALL_SIZE, L.GRAY, weight=500,
            )
        text.queue(f"({hud.verdict_confidence})", (x, y + 286), L.SMALL_SIZE, L.GRAY, weight=500)


def draw_pose_inset(canvas: np.ndarray, pose: Optional[PoseState], text: TextDrawer, min_conf: float) -> None:
    x, y, w, h = L.INSET_RECT
    cv2.rectangle(canvas, (x, y), (x + w, y + h), L.VIOLET, 2)
    cv2.rectangle(canvas, (x - 1, y - 26), (x + w + 1, y), L.VIOLET, -1)
    text.queue("pose at release", (x + 8, y - 13), 18, L.WHITE, weight=600, anchor="lm")
    if pose is None:
        text.queue("(no pose)", (x + 14, y + 40), L.SMALL_SIZE, L.SILVER, weight=400)
        return
    pts = {
        n: (px, py)
        for n, (px, py, c) in pose.keypoints.items()
        if c >= min_conf
    }
    if len(pts) < 4:
        text.queue("(low confidence)", (x + 14, y + 40), L.SMALL_SIZE, L.SILVER, weight=400)
        return
    xs, ys = [p[0] for p in pts.values()], [p[1] for p in pts.values()]
    bw, bh = max(xs) - min(xs) or 1.0, max(ys) - min(ys) or 1.0
    inner_x, inner_y, inner_w, inner_h = x + 24, y + 18, w - 48, h - 36
    s = min(inner_w / bw, inner_h / bh)
    ox = inner_x + (inner_w - bw * s) / 2 - min(xs) * s
    oy = inner_y + (inner_h - bh * s) / 2 - min(ys) * s

    def pt(n: str) -> tuple[int, int]:
        px, py = pts[n]
        return int(px * s + ox), int(py * s + oy)

    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(canvas, pt(a), pt(b), L.VIOLET_LIGHT, 3)
    for n in pts:
        cv2.circle(canvas, pt(n), 5, L.VIOLET_LIGHT, -1)


def segment_angle_signed(a: tuple, b: tuple) -> float:
    """Angle of the (upper joint -> lower joint) segment vs vertical, signed
    by horizontal direction; the reference table's convention."""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))


def draw_joint_table(canvas: np.ndarray, pose: Optional[PoseState], text: TextDrawer, min_conf: float) -> None:
    x, y, w, h = L.TABLE_RECT
    col1, col2, col3 = x + 20, x + w - 460, x + w - 220
    text.queue("Joint Angles", (col1, y + 12), 24, L.VIOLET_TEXT, weight=600)
    text.queue("Left Side", (col2, y + 14), L.TABLE_SIZE, L.GRAY, weight=600)
    text.queue("Right Side", (col3, y + 14), L.TABLE_SIZE, L.GRAY, weight=600)
    cv2.line(canvas, (x, y + 50), (x + w, y + 50), L.LINE, 1)

    def joint(side: str, name: str):
        kp = pose.keypoints.get(f"{side}_{name}") if pose else None
        return (kp[0], kp[1]) if kp and kp[2] >= min_conf else None

    for i, (label, upper, lower) in enumerate(TABLE_ROWS):
        ry = y + 72 + i * 48
        text.queue(label, (col1, ry), L.TABLE_SIZE, L.INK, weight=500)
        for col, side in ((col2, "left"), (col3, "right")):
            a, b = joint(side, upper), joint(side, lower)
            val = f"{segment_angle_signed(a, b):.0f}°" if a and b else "--"
            text.queue(val, (col, ry), L.TABLE_SIZE, L.INK, weight=500)

