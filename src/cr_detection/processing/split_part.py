"""
Screenshot splitting for Clash Royale game state detection.
Ported from KataCR/katacr/build_dataset/utils/split_part.py

Splits a full game screenshot into three parts:
- Part 1: Time display (top right)
- Part 2: Arena/battlefield (main game area)
- Part 3: Card hand (bottom)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple, Dict, Union

from .constants import split_bbox_params, part_sizes, ratio_ranges


def ratio2name(img: Union[np.ndarray, Image.Image]) -> Optional[str]:
    """
    Determine the aspect ratio category of an image.

    Args:
        img: Input image as numpy array or PIL Image.

    Returns:
        Ratio name string (e.g., '2.16', '2.22', '1.78') or None.
    """
    if isinstance(img, Image.Image):
        img = np.array(img)

    r = img.shape[0] / img.shape[1]  # height / width

    for name, ratio in ratio_ranges.items():
        if name == 'part2':
            continue  # Skip internal ratio
        if ratio[0] <= r <= ratio[1]:
            return name

    # If no exact match, find closest
    closest_name = None
    closest_diff = float('inf')
    for name, ratio in ratio_ranges.items():
        if name == 'part2':
            continue
        mid = (ratio[0] + ratio[1]) / 2
        diff = abs(r - mid)
        if diff < closest_diff:
            closest_diff = diff
            closest_name = name

    return closest_name


def extract_bbox(
    image: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extract a bounding box region from an image using proportional coordinates.

    Args:
        image: Input image array.
        x: Left edge as proportion (0.0 to 1.0).
        y: Top edge as proportion (0.0 to 1.0).
        w: Width as proportion.
        h: Height as proportion.
        target_size: Optional (width, height) to resize extracted region.

    Returns:
        Extracted image region.
    """
    shape = image.shape
    if len(shape) == 2:
        image = image[..., None]

    x_px = int(shape[1] * x)
    y_px = int(shape[0] * y)
    w_px = int(shape[1] * w)
    h_px = int(shape[0] * h)

    extracted = image[y_px:y_px + h_px, x_px:x_px + w_px, :]

    if len(shape) == 2:
        extracted = extracted[..., 0]

    if target_size is not None:
        extracted = cv2.resize(extracted, target_size, interpolation=cv2.INTER_CUBIC)

    return extracted


def process_part(
    img: np.ndarray,
    part: Union[int, str],
    playback: bool = False,
    resize: bool = True,
    verbose: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, Tuple]]:
    """
    Extract a specific part from a game screenshot.

    Args:
        img: Full game screenshot as numpy array.
        part: Part number (1, 2, or 3) or string like 'part1'.
        playback: Whether this is a playback/replay screenshot.
        resize: Whether to resize to standard part size.
        verbose: Return bbox parameters along with image.

    Returns:
        Extracted part image, or tuple of (image, bbox_params) if verbose.
    """
    if not isinstance(part, str):
        part = f"part{part}"

    target_size = None
    if resize and part in part_sizes:
        target_size = part_sizes[part]

    if playback:
        part += '_playback'

    # Determine aspect ratio
    ratio_name = ratio2name(img)
    if ratio_name is None:
        ratio_name = '1.78'  # Default to 16:9

    part_key = f"{part}_{ratio_name}"

    # Get bbox parameters
    if part_key not in split_bbox_params:
        # Try without ratio suffix for custom ratios
        base_part = part.split('_')[0]
        # Fall back to closest available ratio
        for key in split_bbox_params:
            if key.startswith(base_part):
                part_key = key
                break

    bbox_params = split_bbox_params.get(part_key)

    if bbox_params is None:
        raise ValueError(f"No bbox params found for {part_key}")

    if isinstance(bbox_params, dict):
        ret = {}
        for key, value in bbox_params.items():
            ret[key] = extract_bbox(img, *value, target_size)
    else:
        ret = extract_bbox(img, *bbox_params, target_size)

    if not verbose:
        return ret
    return ret, bbox_params


def split_screenshot(
    img: np.ndarray,
    resize: bool = True
) -> Dict[str, np.ndarray]:
    """
    Split a full game screenshot into all three parts.

    Args:
        img: Full game screenshot as numpy array.
        resize: Whether to resize to standard part sizes.

    Returns:
        Dictionary with keys 'part1', 'part2', 'part3'.
    """
    return {
        'part1': process_part(img, 1, resize=resize),  # Time
        'part2': process_part(img, 2, resize=resize),  # Arena
        'part3': process_part(img, 3, resize=resize),  # Cards
    }


def process_part3_cards(img_part3: np.ndarray) -> list:
    """
    Extract individual card images from part3.

    Args:
        img_part3: Part3 image (card hand area).

    Returns:
        List of 5 card images (next card + 4 hand cards).
    """
    from .constants import part3_bbox_params

    cards = []
    for param in part3_bbox_params:
        card = extract_bbox(img_part3, *param)
        cards.append(card)
    return cards


def get_image_ratio(img: np.ndarray) -> float:
    """
    Get the aspect ratio (height/width) of an image.

    Args:
        img: Input image as numpy array.

    Returns:
        Aspect ratio as float.
    """
    return img.shape[0] / img.shape[1]


def calibrate_bbox_params(
    img: np.ndarray,
    part: int,
    x: float,
    y: float,
    w: float,
    h: float
) -> np.ndarray:
    """
    Helper to visualize bbox parameters for calibration.

    Args:
        img: Full game screenshot.
        part: Part number (1, 2, or 3).
        x, y, w, h: Proportional bbox parameters.

    Returns:
        Image with bbox rectangle drawn.
    """
    vis = img.copy()
    shape = img.shape

    x_px = int(shape[1] * x)
    y_px = int(shape[0] * y)
    w_px = int(shape[1] * w)
    h_px = int(shape[0] * h)

    colors = {1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 0, 0)}
    color = colors.get(part, (255, 255, 255))

    cv2.rectangle(vis, (x_px, y_px), (x_px + w_px, y_px + h_px), color, 2)
    cv2.putText(vis, f'Part {part}', (x_px, y_px - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return vis
