"""Per-shot metric computation (plan 6.4).

All geometry is 2D image space plus the calibrated scale. Every function
returns None when it cannot answer honestly — no garbage values.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

# A trajectory sample: (frame_index, t, x, y, interpolated)
Sample = tuple[int, float, float, float, bool]


def entry_angle_deg(samples: Sequence[Sample], crossing_x: float) -> Optional[float]:
    """Angle of the trajectory tangent at the rim crossing, degrees from
    horizontal. Fit y = ax^2 + bx + c to the observed centroids from apex to
    crossing; the tangent slope at the crossing x gives the angle.
    """
    observed = [(x, y) for _, _, x, y, interp in samples if not interp]
    if len(observed) < 3:
        return None
    xs = [p[0] for p in observed]
    if max(xs) - min(xs) < 1e-6:
        return None  # straight-down trajectory; parabola in x is degenerate
    # Center x before fitting: keeps the normal equations well-conditioned at
    # image-scale coordinates, and the tangent slope is translation-invariant.
    x0 = sum(xs) / len(xs)
    coeffs = _polyfit2([(x - x0, y) for x, y in observed])
    if coeffs is None:
        return None
    a, b, _ = coeffs
    slope = 2 * a * (crossing_x - x0) + b  # dy/dx in image coords (y down)
    return math.degrees(math.atan(abs(slope)))


def _polyfit2(points: list[tuple[float, float]]) -> Optional[tuple[float, float, float]]:
    """Least-squares quadratic fit via normal equations (no numpy dependency
    in the hot path; n is ~a dozen points)."""
    n = len(points)
    sx = sum(x for x, _ in points)
    sx2 = sum(x**2 for x, _ in points)
    sx3 = sum(x**3 for x, _ in points)
    sx4 = sum(x**4 for x, _ in points)
    sy = sum(y for _, y in points)
    sxy = sum(x * y for x, y in points)
    sx2y = sum(x * x * y for x, y in points)
    # Solve [[sx4,sx3,sx2],[sx3,sx2,sx],[sx2,sx,n]] . [a,b,c] = [sx2y,sxy,sy]
    m = [[sx4, sx3, sx2, sx2y], [sx3, sx2, sx, sxy], [sx2, sx, float(n), sy]]
    return _solve3(m)


def _solve3(m: list[list[float]]) -> Optional[tuple[float, float, float]]:
    """Gaussian elimination with partial pivoting on a 3x4 augmented matrix."""
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-12:
            return None
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(3):
            if r == col:
                continue
            f = m[r][col] / m[col][col]
            m[r] = [v - f * pv for v, pv in zip(m[r], m[col])]
    return tuple(m[i][3] / m[i][i] for i in range(3))


def peak_height_m(
    samples: Sequence[Sample], release_y: float, px_per_m: Optional[float]
) -> Optional[float]:
    """Apex rise above the release point, meters."""
    if px_per_m is None:
        return None
    observed_y = [y for _, _, _, y, interp in samples if not interp]
    if not observed_y:
        return None
    rise_px = release_y - min(observed_y)  # y grows downward
    if rise_px <= 0:
        return None
    return rise_px / px_per_m


def release_velocity_ms(
    samples: Sequence[Sample], release_frame: int, px_per_m: Optional[float], n_frames: int = 5
) -> Optional[float]:
    """Mean ball speed over the first ``n_frames`` post-release frames."""
    if px_per_m is None:
        return None
    window = [s for s in samples if release_frame <= s[0] <= release_frame + n_frames]
    speeds = _segment_speeds(window)
    if not speeds:
        return None
    return (sum(speeds) / len(speeds)) / px_per_m


def _segment_speeds(window: Sequence[Sample]) -> list[float]:
    speeds = []
    for prev, cur in zip(window, window[1:]):
        dt = cur[1] - prev[1]
        if dt <= 0:
            continue
        speeds.append(math.hypot(cur[2] - prev[2], cur[3] - prev[3]) / dt)
    return speeds
