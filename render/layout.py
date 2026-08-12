"""M4 layout geometry and palette.

Structure follows the AA reference (assets/reference/aa-reference-layout.md);
palette and type follow the Roboflow brand: Violet 600 primary, Cool Gray
neutrals, Inter, and detection boxes drawn Roboflow-style (2px, square
corners, filled class+confidence label tab).
"""

CANVAS_W, CANVAS_H = 1920, 1080

# Video pane dominates: 78% of width, 16:9 (owner request — video as large
# as possible while every panel stays readable).
VIDEO_RECT = (0, 0, 1500, 844)  # x, y, w, h

# Compact bottom-left strip under the video
INSET_RECT = (16, 872, 208, 192)  # Pose at Release (tab sits above the box)
TABLE_RECT = (248, 858, 1236, 206)  # Joint Angles

# Slim right rail: three stacked panels
PANEL_X = 1500
PANEL_W = CANVAS_W - PANEL_X
PANEL_H = CANVAS_H // 3
PANEL_PAD = 32

# Roboflow palette, dark mode (BGR) — brand rule: Cool Gray 900 canvas,
# violet accents on dark.
SURFACE = (39, 24, 17)         # Cool Gray 900 #111827 — canvas
SURFACE_ALT = (55, 41, 31)     # Cool Gray 800 #1F2937 — cards
LINE = (81, 65, 55)            # Cool Gray 700 #374151 — hairlines
INK = (251, 250, 249)          # Cool Gray 50 #F9FAFB — big numerals on dark
GRAY = (175, 163, 156)         # Cool Gray 400 #9CA3AF — small text
SILVER = (85, 69, 75)          # Cool Gray 600 #4B5563 — the idle "NO"
VIOLET = (237, 58, 124)        # Violet 600 #7C3AED — primary accent / fills
VIOLET_TEXT = (250, 138, 167)  # Violet 400 #A78BFA — violet type on dark
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

# Type scale (Inter px sizes), sized for the slim rail
HEADER_SIZE = 30
BIG_SIZE = 72
SMALL_SIZE = 20
TABLE_SIZE = 22
COUNTER_SIZE = 24
