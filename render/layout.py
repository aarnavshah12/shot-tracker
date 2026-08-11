"""M4 layout geometry and palette (assets/reference/aa-reference-layout.md).

Canvas 1920x1080: video pane top-left (~55% width), pose inset + joint-angle
table strip below it, three stacked metric panels on the right. Cream panels,
orange headers, big near-black mono numerals, gray small caps.
"""

CANVAS_W, CANVAS_H = 1920, 1080

# Video pane (source scaled to fit, top-left)
VIDEO_RECT = (0, 0, 1056, 594)  # x, y, w, h

# Bottom-left strip under the video
INSET_RECT = (16, 610, 352, 454)  # Pose at Release
TABLE_RECT = (384, 610, 656, 454)  # Joint Angles

# Right column: three stacked panels
PANEL_X = 1056
PANEL_W = CANVAS_W - PANEL_X
PANEL_H = CANVAS_H // 3
PANEL_PAD = 44

# Palette (BGR)
CREAM = (226, 236, 242)  # warm cream, not white
LINE = (35, 32, 30)
ORANGE = (12, 89, 232)
INK = (26, 26, 26)
GRAY = (110, 116, 122)
SILVER = (168, 168, 168)
GREEN = (64, 168, 47)
RED = (37, 48, 217)
WHITE = (245, 245, 245)
BLACK = (12, 12, 12)
PINK = (180, 105, 255)
BALL_BLUE = (255, 120, 40)
TRAIL_ORANGE = (0, 140, 255)
RIM_GREEN = (80, 200, 80)

# Type scale (Menlo px sizes)
HEADER_SIZE = 42
BIG_SIZE = 124
SMALL_SIZE = 28
TABLE_SIZE = 26
COUNTER_SIZE = 26
