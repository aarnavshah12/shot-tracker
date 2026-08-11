"""Session calibration (plan section 5).

Scale: the rim inner diameter is 18 inches (0.4572 m). The median rim-box
width over the first ``calibration_frames`` frames gives pixels-per-meter,
recomputed per session. The same samples give a median rim box used for shot
geometry (the rim doesn't move within a session) and as a fallback when the
detector drops the rim for a frame.

Recovery: if live rim detections steadily contradict the median box
(``drift_frames`` consecutive disagreements), the early samples were garbage
— a cold-start misdetection streak or a camera bump — and calibration resets
and re-collects. Without this, a bad first second of footage would silently
poison scale and rim geometry for the whole session.

Homography (shooter court position) is a session asset supplied by the owner's
4 ground-point clicks; absent homography disables the shot chart only.
"""

from __future__ import annotations

import statistics
from typing import Optional

from engine.types import Detection

Box = tuple[float, float, float, float]


class ScaleCalibration:
    def __init__(
        self,
        rim_diameter_m: float,
        max_samples: int,
        min_samples: int,
        drift_frames: int = 30,
    ):
        self._rim_diameter_m = rim_diameter_m
        self._max_samples = max_samples
        self._min_samples = min_samples
        self._drift_frames = drift_frames
        self._widths: list[float] = []
        self._boxes: list[Box] = []
        self._drift_run = 0

    def observe_rim(self, rim: Detection) -> None:
        if rim.width <= 0:
            return
        median = self.median_rim_box
        if median is not None and _iou(rim.bbox, median) < 0.5:
            self._drift_run += 1
            if self._drift_run >= self._drift_frames:
                # The live rim has steadily contradicted the median: the
                # locked-in samples were wrong, or the camera moved. Restart.
                self._widths.clear()
                self._boxes.clear()
                self._drift_run = 0
            else:
                return  # disagreeing sample: don't poison the median
        else:
            self._drift_run = 0
        if len(self._widths) >= self._max_samples:
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
    def median_rim_box(self) -> Optional[Box]:
        """Per-coordinate median rim box; the stable rim geometry for shot
        logic (detection jitter must not fake crossings)."""
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


def _iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)
