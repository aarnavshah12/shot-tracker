"""Release-frame form metrics from pose (plan 6.4).

Every value is confidence-gated: a joint below the keypoint-confidence
threshold at the release frame nulls the metrics that depend on it — null
over garbage, always (plan 4.2).

Convention: image space, y down. Angles are reported in degrees.
"""

from __future__ import annotations

import math
from typing import Optional

from engine.types import PoseState

Point = tuple[float, float]


def _joint(pose: PoseState, name: str, min_conf: float) -> Optional[Point]:
    kp = pose.keypoints.get(name)
    if kp is None or kp[2] < min_conf:
        return None
    return kp[0], kp[1]


def joint_angle_deg(a: Point, b: Point, c: Point) -> Optional[float]:
    """Interior angle at ``b`` formed by segments b->a and b->c."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cosang))


def shooting_side(pose: PoseState, ball_xy: Point, min_conf: float) -> Optional[str]:
    """'left' or 'right': the side whose wrist is nearest the ball."""
    best, best_dist = None, math.inf
    for side in ("left", "right"):
        wrist = _joint(pose, f"{side}_wrist", min_conf)
        if wrist is None:
            continue
        d = math.hypot(wrist[0] - ball_xy[0], wrist[1] - ball_xy[1])
        if d < best_dist:
            best, best_dist = side, d
    return best


def release_form_metrics(
    pose: Optional[PoseState],
    ball_xy: Optional[Point],
    px_per_m: Optional[float],
    min_conf: float,
) -> dict:
    """elbow_deg / knee_deg / shoulder_hip_deg / release_height_m for the
    release frame; any field that cannot be computed honestly is None."""
    out = {
        "elbow_deg": None,
        "knee_deg": None,
        "shoulder_hip_deg": None,
        "release_height_m": None,
    }
    if pose is None or ball_xy is None:
        return out
    side = shooting_side(pose, ball_xy, min_conf)
    if side is None:
        return out

    shoulder = _joint(pose, f"{side}_shoulder", min_conf)
    elbow = _joint(pose, f"{side}_elbow", min_conf)
    wrist = _joint(pose, f"{side}_wrist", min_conf)
    if shoulder and elbow and wrist:
        out["elbow_deg"] = joint_angle_deg(shoulder, elbow, wrist)

    hip = _joint(pose, f"{side}_hip", min_conf)
    knee = _joint(pose, f"{side}_knee", min_conf)
    ankle = _joint(pose, f"{side}_ankle", min_conf)
    if hip and knee and ankle:
        out["knee_deg"] = joint_angle_deg(hip, knee, ankle)

    # Torso tilt: shoulder-midpoint -> hip-midpoint line vs vertical.
    ls, rs = _joint(pose, "left_shoulder", min_conf), _joint(pose, "right_shoulder", min_conf)
    lh, rh = _joint(pose, "left_hip", min_conf), _joint(pose, "right_hip", min_conf)
    if ls and rs and lh and rh:
        smid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hmid = ((lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2)
        dx, dy = smid[0] - hmid[0], smid[1] - hmid[1]
        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            out["shoulder_hip_deg"] = math.degrees(math.atan2(abs(dx), abs(dy)))

    # Release height: ball above the shooter's grounded foot (lower ankle,
    # i.e. larger y). Approximate ground reference; noted in PROGRESS.
    la, ra = _joint(pose, "left_ankle", min_conf), _joint(pose, "right_ankle", min_conf)
    ankles = [p for p in (la, ra) if p]
    if ankles and px_per_m:
        ground_y = max(p[1] for p in ankles)
        rise_px = ground_y - ball_xy[1]
        if rise_px > 0:
            out["release_height_m"] = rise_px / px_per_m
    return out


def confident_wrists(pose: Optional[PoseState], min_conf: float) -> Optional[list[Point]]:
    """Wrist positions trusted enough for the release-separation rule.
    None means 'no pose information this frame' (fall back to velocity-only);
    an empty list means 'pose seen, wrists not trusted' — also a fallback."""
    if pose is None:
        return None
    wrists = []
    for name in ("left_wrist", "right_wrist"):
        kp = pose.keypoints.get(name)
        if kp is not None and kp[2] >= min_conf:
            wrists.append((kp[0], kp[1]))
    return wrists
