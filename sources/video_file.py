"""VideoFileSource: upload-mode frame source.

A thin wrapper — it yields (frame, t) and nothing more. Any behavior beyond
decoding frames belongs in the engine or the app layer, not here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


class VideoFileSource:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._cap = None

    def __enter__(self) -> "VideoFileSource":
        import cv2  # deferred: the pure-engine tests never need OpenCV

        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise IOError(f"could not open video: {self.path}")
        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.fps < 45:
            # Plan 12: the pipeline must run at 30 fps, but rim-crossing frames
            # may be too sparse for `clean` verdicts.
            import warnings

            warnings.warn(
                f"{self.path.name}: {self.fps:.0f} fps footage — rim-crossing "
                "frames may be too sparse for clean verdicts (60 fps recommended)"
            )
        return self

    def __exit__(self, *exc) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def frames(self) -> Iterator[tuple[object, float]]:
        """Yield (frame, t_seconds); t is frame_index / fps for monotonicity."""
        index = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                return
            yield frame, index / self.fps
            index += 1
