"""
Card detection for Clash Royale hand cards.

Detects the 4 cards in the player's hand from a screenshot.
Based on CR-card-vision-simple-model by teammate.
"""

from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from ultralytics import YOLO

from ..processing.constants import MODELS_DIR


# Hand layout configuration (fractions of width/height)
# These may need adjustment for different screen resolutions
HAND_LAYOUTS = {
    # 540x960 (original)
    '1.78_540': {
        'y_top': 0.82,
        'y_bottom': 0.96,
        'x_left': 0.10,
        'x_right': 0.90,
        'margin_x': 0.08,
        'margin_y': 0.05,
    },
    # 1080x1920 (2x scale)
    '1.78_1080': {
        'y_top': 0.82,
        'y_bottom': 0.96,
        'x_left': 0.20,
        'x_right': 0.90,
        'margin_x': 0.04,
        'margin_y': 0.05,
    },
}

# Default model path
DEFAULT_CARD_MODEL = MODELS_DIR / 'card_detector.pt'


def crop_hand_slots(
    frame: np.ndarray,
    num_slots: int = 4,
    layout: Optional[Dict] = None
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    Crop the 4 card slots from a full screenshot.

    Args:
        frame: Full BGR screenshot
        num_slots: Number of card slots (default 4)
        layout: Layout config dict, or None for auto-detect

    Returns:
        Tuple of (list of cropped images, list of box coordinates)
    """
    h, w = frame.shape[:2]

    # Auto-detect layout based on resolution
    if layout is None:
        if w <= 600:
            layout = HAND_LAYOUTS['1.78_540']
        else:
            layout = HAND_LAYOUTS['1.78_1080']

    # Calculate the hand band region
    band_y1 = int(layout['y_top'] * h)
    band_y2 = int(layout['y_bottom'] * h)

    # Apply vertical margin
    v_margin = int(layout['margin_y'] * (band_y2 - band_y1))
    y1 = band_y1 + v_margin
    y2 = band_y2 - v_margin

    # Horizontal region for all 4 cards
    x_left = int(layout['x_left'] * w)
    x_right = int(layout['x_right'] * w)

    total_width = x_right - x_left
    slot_width = total_width / float(num_slots)

    crops = []
    boxes = []

    for i in range(num_slots):
        seg_x1 = x_left + i * slot_width
        seg_x2 = x_left + (i + 1) * slot_width

        # Shrink horizontally inside this slot
        h_margin = layout['margin_x'] * slot_width
        x1 = int(seg_x1 + h_margin)
        x2 = int(seg_x2 - h_margin)

        # Crop the card
        crop = frame[y1:y2, x1:x2]
        crops.append(crop)
        boxes.append((x1, y1, x2, y2))

    return crops, boxes


class CardDetector:
    """
    Detects cards in the player's hand from a Clash Royale screenshot.

    Usage:
        detector = CardDetector()
        cards = detector.detect(screenshot)
        # Returns: [{"slot": 0, "name": "hog-rider", "confidence": 0.95}, ...]
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_confidence: float = 0.20,
        device: Optional[str] = None
    ):
        """
        Initialize the card detector.

        Args:
            model_path: Path to YOLOv8 model for card detection
            min_confidence: Minimum confidence threshold
            device: Device to run on ('mps', 'cuda', 'cpu', or None for auto)
        """
        if model_path is None:
            model_path = DEFAULT_CARD_MODEL

        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Card detection model not found: {model_path}\n"
                "Copy the trained model to models/card_detector.pt"
            )

        self.model = YOLO(str(model_path))
        self.min_confidence = min_confidence

        # Auto-detect device
        if device is None:
            import torch
            if torch.backends.mps.is_available():
                device = 'mps'
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = device
        self.model.to(device)

        # Warm up on MPS
        if device == 'mps':
            dummy = np.zeros((128, 96, 3), dtype=np.uint8)
            self.model(dummy, verbose=False)

    def _predict_single_card(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Run YOLO on a single cropped card image.

        Returns:
            Tuple of (class_name, confidence)
        """
        results = self.model(img, conf=0.01, verbose=False)[0]

        if len(results.boxes) == 0:
            return "unknown", 0.0

        confs = results.boxes.conf.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy().astype(int)

        best_idx = int(confs.argmax())
        conf = float(confs[best_idx])
        cls_id = int(clss[best_idx])

        name = self.model.names[cls_id]

        if conf < self.min_confidence:
            return "unknown", conf

        return name, conf

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect the cards currently in hand from a full screenshot.

        Args:
            frame: BGR numpy array (full Clash Royale screenshot)

        Returns:
            List of dicts with "slot" (0-3), "name", and "confidence"
        """
        crops, boxes = crop_hand_slots(frame)

        results = []
        for i, crop in enumerate(crops):
            name, conf = self._predict_single_card(crop)
            results.append({
                "slot": i,
                "name": name,
                "confidence": conf,
                "box": boxes[i],  # (x1, y1, x2, y2) in original image coords
            })

        return results

    def detect_with_viz(self, frame: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        Detect cards and return visualization.

        Returns:
            Tuple of (detections, visualization image)
        """
        cards = self.detect(frame)

        # Draw on frame
        vis = frame.copy()
        for card in cards:
            x1, y1, x2, y2 = card['box']
            color = (0, 255, 0) if card['name'] != 'unknown' else (128, 128, 128)

            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{card['name']} {card['confidence']:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        return cards, vis


# Import cv2 for visualization (optional)
try:
    import cv2
except ImportError:
    cv2 = None
