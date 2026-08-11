"""Shooter pose adapter: RF-DETR Keypoint, pretrained COCO-17, zero-shot.

The engine takes any ``PoseModel`` — a callable mapping a frame to an
optional ``PoseState``. This adapter wraps ``RFDETRKeypointPreview`` (a
preview API, pinned in models/registry.py; keypoint ordering verified on real
footage) and keeps every rfdetr detail behind the models/ seam.

One shooter in frame (plan scope): the highest-confidence person wins.
"""

from __future__ import annotations

from typing import Optional

from engine.types import PoseState
from models.registry import COCO_KEYPOINTS


class RFDETRPoseModel:
    def __init__(self, min_person_confidence: float = 0.3):
        from rfdetr import RFDETRKeypointPreview  # deferred: heavy import + weights

        self._model = RFDETRKeypointPreview()
        self._min_conf = min_person_confidence

    def __call__(self, frame: object) -> Optional[PoseState]:
        import cv2
        import numpy as np

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._model.predict(rgb, threshold=self._min_conf)
        if result is None or result.is_empty():
            return None
        person = int(np.asarray(result.detection_confidence).argmax())
        xy = np.asarray(result.xy)[person]
        conf = np.asarray(result.confidence)[person]
        return PoseState(
            keypoints={
                name: (float(x), float(y), float(c))
                for name, (x, y), c in zip(COCO_KEYPOINTS, xy, conf)
            }
        )
