"""Renderer stub (full split-screen layout lands at M4, plan section 8).

Contract: takes (frame, FrameState), returns the composed output frame. All
drawing uses supervision primitives at M4; layout is video left ~55%, three
stacked metric panels right (SHOT SPEED / BALL DISTANCE TO RIM / MAKE?),
pose-at-release inset and joint-angle table on the bottom strip, cream
panels, orange headers, big mono numerals per the reference screenshots in
assets/reference/.

No shot logic lives here — if a verdict or a metric isn't already on the
FrameState, the fix goes in the engine, not in drawing code.
"""

from __future__ import annotations

from engine.types import FrameState


class Renderer:
    def __init__(self, config=None):
        self.config = config

    def draw(self, frame: object, state: FrameState) -> object:
        # M4: compose the split-screen layout. Until then, pass-through.
        return frame
