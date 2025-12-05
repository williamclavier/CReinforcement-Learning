"""
Apple Vision framework OCR for Clash Royale.

Uses native macOS Vision framework for fast, reliable text recognition.
"""

import re
from typing import Optional, Tuple

import numpy as np

# Import Vision framework
try:
    import Vision
    import Quartz
    from Foundation import NSData
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False


def image_to_cgimage(image: np.ndarray) -> 'Quartz.CGImageRef':
    """Convert numpy array (BGR) to CGImage."""
    import cv2

    # Convert BGR to RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Create CGImage from raw data
    bytes_per_row = w * 3
    data_provider = Quartz.CGDataProviderCreateWithData(
        None, rgb.tobytes(), h * bytes_per_row, None
    )

    cgimage = Quartz.CGImageCreate(
        w, h,
        8,  # bits per component
        24,  # bits per pixel
        bytes_per_row,
        Quartz.CGColorSpaceCreateDeviceRGB(),
        Quartz.kCGBitmapByteOrderDefault,
        data_provider,
        None,
        False,
        Quartz.kCGRenderingIntentDefault
    )

    return cgimage


def recognize_text(image: np.ndarray) -> str:
    """
    Recognize text in image using Apple Vision.

    Args:
        image: BGR numpy array

    Returns:
        Recognized text string
    """
    if not VISION_AVAILABLE:
        raise RuntimeError("Vision framework not available")

    cgimage = image_to_cgimage(image)

    # Create request handler
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        cgimage, None
    )

    # Create text recognition request
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(False)  # Faster, we just need numbers

    # Perform request
    success, error = handler.performRequests_error_([request], None)

    if not success:
        return ""

    # Get results
    results = request.results()
    if not results:
        return ""

    # Combine all recognized text
    texts = []
    for observation in results:
        candidates = observation.topCandidates_(1)
        if candidates:
            texts.append(candidates[0].string())

    return " ".join(texts)


def parse_game_time(text: str) -> float:
    """
    Parse game time from OCR text.

    Expects format like "2:45" or "2.45" and returns seconds.
    """
    # Try to find time pattern (M:SS or M.SS)
    match = re.search(r'(\d):(\d{2})', text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds

    # Try single number (seconds only)
    match = re.search(r'(\d{1,3})', text)
    if match:
        return float(match.group(1))

    return 0.0


class VisionOCR:
    """
    OCR using Apple Vision framework.

    Fast and accurate for reading game timer and numbers.
    """

    def __init__(self):
        if not VISION_AVAILABLE:
            raise RuntimeError(
                "Apple Vision framework not available. "
                "Install pyobjc-framework-Vision: pip install pyobjc-framework-Vision"
            )

    def process_part1(self, image: np.ndarray, pil: bool = False) -> float:
        """
        Process part1 (timer area) and return game time in seconds.

        Args:
            image: Timer region image (BGR numpy array)
            pil: If True, image is RGB format

        Returns:
            Game time in seconds
        """
        if pil:
            import cv2
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        text = recognize_text(image)
        return parse_game_time(text)

    def read_text(self, image: np.ndarray) -> str:
        """
        Read all text from image.

        Args:
            image: BGR numpy array

        Returns:
            Recognized text
        """
        return recognize_text(image)


def create_vision_ocr() -> Optional[VisionOCR]:
    """Create VisionOCR instance, or None if not available."""
    try:
        return VisionOCR()
    except Exception as e:
        print(f"Vision OCR not available: {e}")
        return None
