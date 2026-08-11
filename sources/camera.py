"""CameraSource: live-mode frame source.

Deliberately unimplemented until the engine passes M4 (plan rule: no
camera-feed code before then). It exists now only to pin the interface —
identical to VideoFileSource: a context manager with a frames() iterator of
(frame, t). Live mode must differ from upload mode in nothing but this
wrapper and model-size config.
"""

from __future__ import annotations

from typing import Iterator


class CameraSource:
    def __init__(self, device_index: int = 0):
        raise NotImplementedError("Live mode is M5; no camera-feed code before the engine passes M4.")

    def __enter__(self) -> "CameraSource":
        raise NotImplementedError

    def __exit__(self, *exc) -> None:
        raise NotImplementedError

    def frames(self) -> Iterator[tuple[object, float]]:
        raise NotImplementedError
