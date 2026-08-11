"""Batched text rendering: queue draws during composition, rasterize in one
PIL pass per frame (Menlo — the reference's mono look; Hershey fallback when
no TTF is available)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


class TextDrawer:
    def __init__(self):
        self._queue: list[tuple] = []
        self._fonts: dict[tuple[int, int], object] = {}
        self._font_path = next((p for p in _FONT_CANDIDATES if Path(p).exists()), None)

    def _font(self, size: int, stroke: int):
        key = (size, stroke)
        if key not in self._fonts:
            from PIL import ImageFont

            self._fonts[key] = ImageFont.truetype(self._font_path, size)
        return self._fonts[key]

    def queue(
        self,
        text: str,
        xy: tuple[int, int],
        size: int,
        color_bgr: tuple[int, int, int],
        stroke: int = 0,
        anchor: str = "la",
    ) -> None:
        self._queue.append((text, xy, size, color_bgr, stroke, anchor))

    def flush(self, canvas_bgr: np.ndarray) -> np.ndarray:
        if not self._queue:
            return canvas_bgr
        if self._font_path is None:
            return self._flush_cv2(canvas_bgr)
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas_bgr[:, :, ::-1])
        draw = ImageDraw.Draw(img)
        for text, xy, size, (b, g, r), stroke, anchor in self._queue:
            draw.text(
                xy, text, font=self._font(size, stroke), fill=(r, g, b),
                stroke_width=stroke, stroke_fill=(r, g, b), anchor=anchor,
            )
        self._queue.clear()
        return np.asarray(img)[:, :, ::-1].copy()

    def _flush_cv2(self, canvas: np.ndarray) -> np.ndarray:
        import cv2

        for text, (x, y), size, color, stroke, anchor in self._queue:
            scale = size / 28.0
            if anchor and "s" in anchor:  # baseline anchors used by callers
                pass
            cv2.putText(
                canvas, text, (int(x), int(y + size * 0.8)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, max(1, 1 + stroke),
            )
        self._queue.clear()
        return canvas
