"""M4 layout geometry and palette.

Structure follows the AA reference (assets/reference/aa-reference-layout.md);
palette and type follow the Roboflow brand: Violet 600 primary, Cool Gray
neutrals, Inter, and detection boxes drawn Roboflow-style (2px, square
corners, filled class+confidence label tab).
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

# Roboflow palette (BGR)
SURFACE = (251, 250, 249)      # Cool Gray 50  #F9FAFB
SURFACE_ALT = (246, 244, 243)  # Cool Gray 100 #F3F4F6
LINE = (219, 213, 209)         # Cool Gray 300 #D1D5DB
INK = (39, 24, 17)             # Cool Gray 900 #111827
GRAY = (128, 114, 107)         # Cool Gray 500 #6B7280
SILVER = (175, 163, 156)       # Cool Gray 400 #9CA3AF
VIOLET = (237, 58, 124)        # Violet 600 #7C3AED — primary accent
VIOLET_LIGHT = (253, 181, 196) # Violet 300 #C4B5FD
GREEN = (129, 185, 16)         # Emerald 500 #10B981
RED = (68, 68, 239)            # Red 500 #EF4444
CYAN = (178, 145, 8)           # Cyan 600 #0891B2
AMBER = (11, 158, 245)         # Amber 500 #F59E0B
WHITE = (255, 255, 255)

# Video-overlay class colors (categorical palette; Violet 600 is class 1)
BALL_COLOR = VIOLET
RIM_COLOR = CYAN
TRAIL_COLOR = AMBER
SKELETON_COLOR = VIOLET_LIGHT
TAG_COLOR = VIOLET

# Type scale (Inter px sizes)
HEADER_SIZE = 40
BIG_SIZE = 124
SMALL_SIZE = 28
TABLE_SIZE = 26
COUNTER_SIZE = 26
