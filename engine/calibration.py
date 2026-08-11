"""Session calibration (plan section 5).

Scale: the rim inner diameter is 18 inches (0.4572 m). The median rim-box
width over the first ``calibration_frames`` frames gives pixels-per-meter,
recomputed per session. The same samples give a median rim box used as a
fallback when the detector drops the rim for a frame.

Homography (shooter court position) is a session asset supplied by the owner's
4 ground-point clicks; absent homography disables the shot chart only.
"""

from __future__ import annotations

import statistics
from typing import Optional

from engine.types import Detection


class ScaleCalibration:
    def __init__(self, rim_diameter_m: float, max_samples: int, min_samples: int):
        self._rim_diameter_m = rim_diameter_m
        self._max_samples = max_samples
        self._min_samples = min_samples
        self._widths: list[float] = []
        self._boxes: list[tuple[float, float, float, float]] = []

    def observe_rim(self, rim: Detection) -> None:
        if len(self._widths) >= self._max_samples:
            return
        if rim.width <= 0:
            return
        self._widths.append(rim.width)
        self._boxes.append(rim.bbox)

    @property
    def px_per_m(self) -> Optional[float]:
        """Pixels per meter, or None until enough rim samples exist."""
        if len(self._widths) < self._min_samples:
            return None
        return statistics.median(self._widths) / self._rim_diameter_m

    @property
    def median_rim_box(self) -> Optional[tuple[float, float, float, float]]:
        """Per-coordinate median rim box; rim-geometry fallback for frames
        where the detector misses the rim (the rim doesn't move in a session)."""
        if len(self._boxes) < self._min_samples:
            return None
        return tuple(statistics.median(b[i] for b in self._boxes) for i in range(4))

    def to_metadata(self) -> dict:
        """Session-metadata payload (plan: cache scale per session)."""
        return {
            "rim_diameter_m": self._rim_diameter_m,
            "samples": len(self._widths),
            "px_per_m": self.px_per_m,
            "median_rim_box": list(self.median_rim_box) if self.median_rim_box else None,
        }
