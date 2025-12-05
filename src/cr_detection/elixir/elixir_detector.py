"""
Elixir detection for Clash Royale using pixel color checking.

Detects elixir count by checking if each bar segment is the purple "filled" color.
"""

import json
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

# Elixir bar purple color (filled) in HSV
# Hue ~149, Saturation 210-230, Value 190-220
ELIXIR_HSV_MIN = np.array([140, 200, 180])
ELIXIR_HSV_MAX = np.array([160, 240, 220])

# Default calibration points (relative coordinates 0-1)
# These can be overridden by elixir_calibration.json
DEFAULT_RELATIVE_POINTS = [
    (0.084, 0.983),
    (0.127, 0.983),
    (0.171, 0.983),
    (0.214, 0.983),
    (0.257, 0.983),
    (0.300, 0.983),
    (0.344, 0.983),
    (0.387, 0.983),
    (0.430, 0.983),
    (0.474, 0.983),
]


class ElixirDetector:
    """
    Detects elixir count by checking pixel colors at calibrated points.
    """

    def __init__(self, calibration_path: Optional[Path] = None):
        """
        Initialize elixir detector.

        Args:
            calibration_path: Path to elixir_calibration.json file
        """
        self.relative_points = DEFAULT_RELATIVE_POINTS

        # Try to load calibration
        if calibration_path is None:
            calibration_path = Path('elixir_calibration.json')

        if calibration_path.exists():
            try:
                with open(calibration_path, 'r') as f:
                    config = json.load(f)
                self.relative_points = [tuple(p) for p in config['relative_points']]
                print(f"Loaded elixir calibration from {calibration_path}")
            except Exception as e:
                print(f"Could not load elixir calibration: {e}")

    def _is_elixir_filled(self, pixel_bgr: np.ndarray) -> bool:
        """Check if a pixel is the purple elixir color."""
        # Convert single pixel to HSV
        pixel_hsv = cv2.cvtColor(
            pixel_bgr.reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2HSV
        )[0, 0]

        # Check if within purple range
        return (
            ELIXIR_HSV_MIN[0] <= pixel_hsv[0] <= ELIXIR_HSV_MAX[0] and
            ELIXIR_HSV_MIN[1] <= pixel_hsv[1] <= ELIXIR_HSV_MAX[1] and
            ELIXIR_HSV_MIN[2] <= pixel_hsv[2] <= ELIXIR_HSV_MAX[2]
        )

    def detect(self, image: np.ndarray) -> int:
        """
        Detect current elixir count from screenshot.

        Args:
            image: BGR numpy array (full screenshot)

        Returns:
            Elixir count (0-10)
        """
        h, w = image.shape[:2]

        # Check all segments and find the rightmost filled one
        # This handles the pulsing animation at 10 elixir where
        # early segments may temporarily appear unfilled
        filled = []
        for rx, ry in self.relative_points:
            x = int(rx * w)
            y = int(ry * h)
            pixel = image[y, x]
            filled.append(self._is_elixir_filled(pixel))

        # Find the highest filled segment (rightmost)
        # If segment N is filled, we have at least N elixir
        elixir_count = 0
        for i in range(len(filled) - 1, -1, -1):  # Check right to left
            if filled[i]:
                elixir_count = i + 1
                break

        return elixir_count

    def detect_with_debug(self, image: np.ndarray) -> Tuple[int, List[bool]]:
        """
        Detect elixir with debug info.

        Returns:
            Tuple of (elixir_count, list of filled states for each segment)
        """
        h, w = image.shape[:2]

        filled = []
        for rx, ry in self.relative_points:
            x = int(rx * w)
            y = int(ry * h)
            pixel = image[y, x]
            filled.append(self._is_elixir_filled(pixel))

        # Count consecutive filled from start
        elixir_count = 0
        for is_filled in filled:
            if is_filled:
                elixir_count += 1
            else:
                break

        return elixir_count, filled


def create_elixir_detector() -> ElixirDetector:
    """Create an ElixirDetector instance."""
    return ElixirDetector()
