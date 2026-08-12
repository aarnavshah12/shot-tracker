"""Batched text rendering: queue draws during composition, rasterize in one
PIL pass per frame. Inter (variable weights, the Roboflow brand face) with a
Menlo/Hershey fallback chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_INTER = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "Inter-Variable.ttf"
_FALLBACKS = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


class TextDrawer:
    def __init__(self):
        self._queue: list[tuple] = []
        self._fonts: dict[tuple[int, int], object] = {}
        if _INTER.exists():
            self._font_path, self._variable = str(_INTER), True
        else:
            self._font_path = next((p for p in _FALLBACKS if Path(p).exists()), None)
            self._variable = False

    def _font(self, size: int, weight: int):
        key = (size, weight)
        if key not in self._fonts:
            from PIL import ImageFont

            font = ImageFont.truetype(self._font_path, size)
            if self._variable:
                try:
                    font.set_variation_by_axes([14, weight])  # opsz, wght
                except Exception:
                    pass
            self._fonts[key] = font
        return self._fonts[key]

    def queue(
        self,
        text: str,
        xy: tuple[int, int],
        size: int,
        color_bgr: tuple[int, int, int],
        weight: int = 400,
        anchor: str = "la",
    ) -> None:
        self._queue.append((text, xy, size, color_bgr, weight, anchor))

    def flush(self, canvas_bgr: np.ndarray) -> np.ndarray:
        if not self._queue:
            return canvas_bgr
        if self._font_path is None:
            return self._flush_cv2(canvas_bgr)
        from PIL import Image, ImageDraw

        img = Image.fromarray(canvas_bgr[:, :, ::-1])
        draw = ImageDraw.Draw(img)
        for text, xy, size, (b, g, r), weight, anchor in self._queue:
            draw.text(xy, text, font=self._font(size, weight), fill=(r, g, b), anchor=anchor)
        self._queue.clear()
        return np.asarray(img)[:, :, ::-1].copy()

    def _flush_cv2(self, canvas: np.ndarray) -> np.ndarray:
        import cv2

        for text, (x, y), size, color, weight, anchor in self._queue:
            scale = size / 28.0
            cv2.putText(
                canvas, text, (int(x), int(y + size * 0.8)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2 if weight >= 600 else 1,
            )
        self._queue.clear()
        return canvas
