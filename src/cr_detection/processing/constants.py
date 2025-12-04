"""
Configuration constants for Clash Royale screenshot processing.
Includes split parameters for different screen aspect ratios.
"""

from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parents[3]
MODELS_DIR = ROOT_DIR / "models"

# Standard processed image sizes
part_sizes = {
    'part1': (97, 48),     # Time display area
    'part2': (576, 896),   # Arena/battlefield
    'part3': (600, 200),   # Card hand area
}

# Arena size for coordinate calculations
background_size = (576, 896)

# Split bounding box parameters for different aspect ratios
# Format: [x_top_left, y_top_left, width, height] as proportions (0.0 to 1.0)
split_bbox_params = {
    # 2.16 ratio (592x1280) - Original KataCR format
    'part1_2.16': (0.835, 0.063, 0.165, 0.038),   # Time area - top right
    'part2_2.16': (0.021, 0.073, 0.960, 0.700),   # Arena/battlefield
    'part3_2.16': (0.000, 0.808, 1.000, 0.155),   # Card hand area
    'part4_2.16': {
        'up': (0.100, 0.340, 0.800, 0.070),
        'mid': (0.180, 0.410, 0.650, 0.050),
    },

    # 2.22 ratio (1080x2400 or 576x1280)
    'part1_2.22': (0.835, 0.058, 0.165, 0.038),
    'part2_2.22': (0.020, 0.070, 0.960, 0.690),
    'part2_playback_2.22': (0.024, 0.196, 0.954, 0.685),
    'part3_2.22': (0.000, 0.850, 1.000, 0.150),
    'part4_2.22': {
        'up': (0.130, 0.352, 0.747, 0.051),
    },

    # 2.13 ratio (600x1280)
    'part1_2.13': (0.845, 0.037, 0.165, 0.038),
    'part2_2.13': (0.026, 0.048, 0.960, 0.710),
    'part3_2.13': (0.000, 0.845, 1.000, 0.160),

    # 1.78 ratio (1080x1920 - Bluestacks 16:9) - NEW
    # These are PLACEHOLDER values that need calibration with actual screenshots!
    # TODO: Calibrate these values with real Bluestacks screenshots
    'part1_1.78': (0.835, 0.020, 0.165, 0.045),   # Time area - top right (smaller due to shorter screen)
    'part2_1.78': (0.020, 0.050, 0.960, 0.750),   # Arena/battlefield (larger proportion)
    'part3_1.78': (0.000, 0.820, 1.000, 0.180),   # Card hand area
}

# Aspect ratio ranges for auto-detection
ratio_ranges = {
    'part2': (1.57, 1.58),   # Arena aspect ratio
    '2.16': (2.16, 2.17),    # 592x1280
    '2.22': (2.22, 2.23),    # 1080x2400 or 576x1280
    '2.13': (2.13, 2.14),    # 600x1280
    '1.78': (1.77, 1.79),    # 1080x1920 (16:9 Bluestacks)
}

# Card positions in part3 (for future hand detection)
part3_bbox_params = [
    (0.047, 0.590, 0.100, 0.365),  # next card
    (0.222, 0.000, 0.185, 0.745),  # card1
    (0.410, 0.000, 0.185, 0.745),  # card2
    (0.600, 0.000, 0.185, 0.745),  # card3
    (0.785, 0.000, 0.185, 0.745),  # card4
]

# Elixir position in part3
part3_elixir_params = (0.262, 0.700, 0.067, 0.160)

# Detection thresholds
DETECTION_CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.6
OCR_CONF_THRESHOLD = 0.9

# State builder constants
BAR_CENTER2BODY_DELTA_Y = 40
DIS_BAR_AND_BAR_LEVEL_THRE = 15
DIS_BAR_AND_BODY_THRE = 35
