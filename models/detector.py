"""Ball + rim detector adapter.

The engine takes any ``Detector`` — a callable mapping a frame to a list of
``Detection``. Tests inject scripted callables; production uses
``RFDETRDetector`` below, which is wired up at M1 once the bootstrap model
finishes training.
"""

from __future__ import annotations

from typing import Callable, Protocol

from engine.types import Detection

Detector = Callable[[object], list[Detection]]


class DetectorProtocol(Protocol):
    def __call__(self, frame: object) -> list[Detection]: ...


class RFDETRDetector:
    """Loads detector v0 by model ID and adapts its output to ``Detection``.

    Implemented at M1 (the model is still training). Keeping the class here
    pins the adapter seam: model-API details never leak past models/.
    """

    def __init__(self, model_id: str, ball_confidence: float, rim_confidence: float):
        self.model_id = model_id
        self.ball_confidence = ball_confidence
        self.rim_confidence = rim_confidence

    def __call__(self, frame: object) -> list[Detection]:
        raise NotImplementedError(
            "M1: load detector v0 by model ID once "
            "aarnavs-space/basketball-shooting-robot-kbsro-1-rfdetr-small-t1 "
            "finishes training. Filter to DETECTOR_CLASSES and per-class confidence."
        )
