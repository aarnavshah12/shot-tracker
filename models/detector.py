"""Ball + rim detector adapter.

The engine takes any ``Detector`` — a callable mapping a frame to a list of
``Detection``. Tests inject scripted callables; production uses
``RFDETRDetector`` below: detector v0 loaded by model ID through the
``inference`` runtime (local ONNX execution; the hosted weights download on
first load). Model-API details never leak past models/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Protocol

from engine.types import Detection
from models.registry import DETECTOR_CLASSES

Detector = Callable[[object], list[Detection]]


class DetectorProtocol(Protocol):
    def __call__(self, frame: object) -> list[Detection]: ...


class RFDETRDetector:
    """Detector v0 (RF-DETR Small bootstrap) behind the engine's callable seam.

    The source dataset carries extra classes (person, people, shoot, made);
    only ``DETECTOR_CLASSES`` pass through, each gated by its own confidence
    threshold. Everything else the model says is dropped here, silently.
    """

    def __init__(
        self,
        model_id: str,
        ball_confidence: float = 0.4,
        rim_confidence: float = 0.4,
        api_key: Optional[str] = None,
    ):
        from inference import get_model  # deferred: heavy import, model download

        self.model_id = model_id
        self._thresholds = {"ball": ball_confidence, "rim": rim_confidence}
        self._floor = min(self._thresholds.values())
        self._model = get_model(model_id, api_key=api_key or _api_key())

    def __call__(self, frame: object) -> list[Detection]:
        result = self._model.infer(frame, confidence=self._floor)[0]
        detections = []
        for p in result.predictions:
            name = p.class_name
            if name not in DETECTOR_CLASSES or p.confidence < self._thresholds[name]:
                continue
            half_w, half_h = p.width / 2, p.height / 2
            detections.append(
                Detection(
                    class_name=name,
                    confidence=p.confidence,
                    bbox=(p.x - half_w, p.y - half_h, p.x + half_w, p.y + half_h),
                )
            )
        return detections


def _api_key() -> str:
    """ROBOFLOW_API_KEY from the environment, falling back to the repo .env
    (gitignored). The key never appears in code or logs."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.strip().partition("=")
            if name == "ROBOFLOW_API_KEY" and value:
                return value
    raise RuntimeError(
        "ROBOFLOW_API_KEY not set: export it or put it in the repo's .env"
    )
