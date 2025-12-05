"""
High-level GameStateDetector API for Clash Royale.

Combines all detection components into a single easy-to-use interface.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union

from ..capture.bluestacks import BluestacksCapture, save_screenshot
from ..models.yolo_detector import ComboDetector, create_detector
from ..ocr.paddle_ocr import OCR, create_ocr
from ..processing.split_part import process_part, split_screenshot
from .state_builder import StateBuilder, GameState, UnitInfo, CardInfo


class GameStateDetector:
    """
    High-level API for detecting Clash Royale game state.

    Combines screenshot capture, YOLO detection, OCR, and state building
    into a single unified interface.

    Example:
        ```python
        detector = GameStateDetector()

        # Capture and detect from Bluestacks
        state = detector.capture_and_detect()
        print(f"Time: {state.time}s")
        print(f"Friendly units: {len(state.get_friendly_units())}")
        print(f"Enemy units: {len(state.get_enemy_units())}")

        # Or detect from an image file
        state = detector.detect_from_file("screenshot.png")
        ```
    """

    def __init__(
        self,
        model_paths: Optional[List[str]] = None,
        use_tracking: bool = True,
        use_gpu: bool = True,
        use_quartz: bool = True,
        use_ocr: bool = True,
        use_cards: bool = True
    ):
        """
        Initialize the game state detector.

        Args:
            model_paths: Optional list of YOLOv8 model paths.
            use_tracking: Enable ByteTrack object tracking.
            use_gpu: Use GPU for inference.
            use_quartz: Use Quartz for macOS screenshot capture.
            use_ocr: Enable OCR for time detection (can be slow to initialize).
            use_cards: Enable card detection for hand cards.
        """
        # Initialize components
        self.capture = BluestacksCapture(use_quartz=use_quartz)
        self.detector = create_detector(model_paths, use_tracking=use_tracking)
        self.state_builder = StateBuilder(ocr=None)  # OCR handled separately

        # OCR for timer (prefer Apple Vision, fallback to PaddleOCR)
        self.ocr = None
        if use_ocr:
            try:
                from ..ocr.vision_ocr import VisionOCR
                self.ocr = VisionOCR()
                print("Vision OCR initialized!")
            except Exception as e:
                print(f"Vision OCR failed ({e}), trying PaddleOCR...")
                try:
                    self.ocr = create_ocr(use_gpu=use_gpu)
                    print("PaddleOCR initialized!")
                except Exception as e2:
                    print(f"OCR disabled: {e2}")

        # Card detector (optional)
        self.card_detector = None
        if use_cards:
            try:
                from ..cards.card_detector import CardDetector
                self.card_detector = CardDetector()
                print("Card detector initialized!")
            except Exception as e:
                print(f"Card detection disabled: {e}")

        self._last_screenshot: Optional[np.ndarray] = None
        self._last_arena: Optional[np.ndarray] = None  # part2 - where detections happen
        self._last_state: Optional[GameState] = None

    def capture_and_detect(self) -> Optional[GameState]:
        """
        Capture a screenshot from Bluestacks and detect game state.

        Returns:
            GameState object, or None if capture fails.
        """
        screenshot = self.capture.capture()
        if screenshot is None:
            return None

        return self.detect_from_image(screenshot)

    def detect_from_image(self, image: np.ndarray, rgb: bool = False) -> GameState:
        """
        Detect game state from an image.

        Args:
            image: Input image as numpy array.
            rgb: If True, image is RGB. If False, image is BGR.

        Returns:
            GameState object.
        """
        if rgb:
            image = image[..., ::-1]  # RGB -> BGR

        self._last_screenshot = image

        # Split screenshot into parts
        parts = split_screenshot(image)
        self._last_arena = parts['part2']

        # Detect time from timer region (top right of screen)
        time = 0.0
        if self.ocr is not None:
            try:
                h, w = image.shape[:2]
                timer_region = image[0:int(h*0.09), int(w*0.6):int(w*0.95)]
                time = self.ocr.process_part1(timer_region, pil=False)
            except Exception:
                pass

        # Detect units from part2 (arena)
        arena_result = self.detector.detect(self._last_arena, rgb=False)

        # Build state
        state = self.state_builder.update(arena_result, time=time)

        # Detect cards in hand (uses full screenshot)
        if self.card_detector is not None:
            try:
                card_results = self.card_detector.detect(image)
                for card in card_results:
                    state.cards.append(CardInfo(
                        slot=card['slot'],
                        name=card['name'],
                        confidence=card['confidence']
                    ))
            except Exception as e:
                pass  # Card detection failed, continue without cards

        self._last_state = state

        return state

    def detect_from_file(self, path: Union[str, Path]) -> GameState:
        """
        Detect game state from an image file.

        Args:
            path: Path to image file.

        Returns:
            GameState object.
        """
        import cv2
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        return self.detect_from_image(image, rgb=False)

    def get_last_screenshot(self) -> Optional[np.ndarray]:
        """Get the last processed screenshot."""
        return self._last_screenshot

    def get_last_arena(self) -> Optional[np.ndarray]:
        """Get the last arena image (part2) where detections are made."""
        return self._last_arena

    def get_last_state(self) -> Optional[GameState]:
        """Get the last detected game state."""
        return self._last_state

    def get_visualization(self, show_boxes: bool = True) -> Optional[np.ndarray]:
        """
        Get a visualization of the last detection.

        Args:
            show_boxes: Draw detection boxes on the image.

        Returns:
            Visualization image in BGR format.
        """
        if self._last_screenshot is None:
            return None

        if show_boxes and self.detector.result is not None:
            parts = split_screenshot(self._last_screenshot)
            vis = self.detector.result.show_box(show_conf=True)
            return vis

        return self._last_screenshot.copy()

    def save_visualization(self, path: Union[str, Path]) -> bool:
        """
        Save the current visualization to a file.

        Args:
            path: Output file path.

        Returns:
            True if successful.
        """
        vis = self.get_visualization()
        if vis is None:
            return False
        return save_screenshot(vis, str(path))

    def reset(self):
        """Reset the detector state (clear tracking, memory, etc.)."""
        self.detector.reset_tracker()
        self.state_builder.reset()
        self._last_screenshot = None
        self._last_state = None

    def get_detections(self) -> Dict:
        """
        Get raw detection data from the last frame.

        Returns:
            Dictionary with detection information.
        """
        return self.detector.get_detections()


def create_game_state_detector(
    use_tracking: bool = True,
    use_gpu: bool = True
) -> GameStateDetector:
    """
    Factory function to create a GameStateDetector.

    Args:
        use_tracking: Enable object tracking.
        use_gpu: Use GPU acceleration.

    Returns:
        Configured GameStateDetector instance.
    """
    return GameStateDetector(
        use_tracking=use_tracking,
        use_gpu=use_gpu
    )
