"""
Utility functions for Clash Royale game state detection.
Ported from KataCR/katacr/policy/perceptron/utils.py
"""

import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from typing import Tuple, Union, Optional

# Lowercase alphabet for text matching
LOW_ALPHA = [chr(ord('a') + i) for i in range(26)]

# Arena grid configuration
background_size = (576, 896)  # Arena image size
xyxy_grids = (4, 82, 574, 884)  # Grid bounds in pixels
grid_size = (18, 32)  # Grid dimensions (x, y)

# Cell size in pixels (width, height)
cell_size = np.array([
    (xyxy_grids[2] - xyxy_grids[0]) / grid_size[0],
    (xyxy_grids[3] - xyxy_grids[1]) / grid_size[1]
], dtype=np.float32)


def extract_img(
    img: np.ndarray,
    xyxy: Union[list, np.ndarray],
    target_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Extract a rectangular region from an image.

    Args:
        img: Source image.
        xyxy: [x1, y1, x2, y2] coordinates.
        target_size: Optional (width, height) to resize.

    Returns:
        Extracted image region.
    """
    xyxy = np.array(xyxy, np.int32)
    extracted = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
    if target_size is not None:
        extracted = cv2.resize(extracted, target_size, interpolation=cv2.INTER_CUBIC)
    return extracted


def cell2pixel(xy: Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert cell coordinates to pixel coordinates.

    Args:
        xy: Cell coordinates (x, y).

    Returns:
        Pixel coordinates as int32 array.
    """
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)
    return (xy * cell_size + xyxy_grids[:2]).astype(np.int32)


def pixel2cell(xy: Union[list, np.ndarray]) -> np.ndarray:
    """
    Convert pixel coordinates to cell coordinates.

    Args:
        xy: Pixel coordinates (x, y).

    Returns:
        Cell coordinates as float32 array.
    """
    if not isinstance(xy, np.ndarray):
        xy = np.array(xy)
    return ((xy - xyxy_grids[:2]) / cell_size).astype(np.float32)


def xyxy2center(xyxy: Union[list, np.ndarray]) -> np.ndarray:
    """
    Get the center point of a bounding box.

    Args:
        xyxy: [x1, y1, x2, y2] coordinates.

    Returns:
        Center point [cx, cy].
    """
    return np.array([
        (xyxy[0] + xyxy[2]) / 2,
        (xyxy[1] + xyxy[3]) / 2
    ], np.float32)


def xyxy2topcenter(xyxy: Union[list, np.ndarray]) -> np.ndarray:
    """
    Get the top center point of a bounding box.

    Args:
        xyxy: [x1, y1, x2, y2] coordinates.

    Returns:
        Top center point [cx, y1].
    """
    return np.array([
        (xyxy[0] + xyxy[2]) / 2,
        xyxy[1]
    ], np.float32)


def xyxy2sub(
    xyxy: Union[list, np.ndarray],
    sub: Union[list, np.ndarray]
) -> np.ndarray:
    """
    Get a sub-region of a bounding box using proportional coordinates.

    Args:
        xyxy: [x1, y1, x2, y2] parent coordinates.
        sub: [x, y, x2, y2] proportional sub-coordinates (0.0 to 1.0).

    Returns:
        Absolute coordinates of sub-region.
    """
    xyxy = np.array(xyxy)
    w, h = xyxy[2:] - xyxy[:2]
    if not isinstance(sub, np.ndarray):
        sub = np.array(sub)
    delta = sub * np.array([w, h, w, h])
    return np.concatenate([xyxy[:2], xyxy[:2]]) + delta


def edit_distance(s1: str, s2: str, dis: Optional[str] = None) -> int:
    """
    Compute Levenshtein distance between two strings.

    Args:
        s1: First string.
        s2: Second string.
        dis: If 's1', return min distance where s1 is substring.
             If 's2', return min distance where s2 is substring.

    Returns:
        Edit distance as integer.
    """
    if dis == 's2':
        s1, s2 = s2, s1

    m = len(s1) + 1
    n = len(s2) + 1
    s1_dis = max(n, m)

    tbl = {}
    for i in range(m):
        tbl[i, 0] = i
    for j in range(n):
        tbl[0, j] = j

    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            tbl[i, j] = min(
                tbl[i, j - 1] + 1,
                tbl[i - 1, j] + 1,
                tbl[i - 1, j - 1] + cost
            )
            if i == m - 1:
                s1_dis = min(s1_dis, tbl[i, j])

    if dis == 's1' or dis == 's2':
        return s1_dis
    return tbl[i, j]


def pil_draw_text(
    img: Union[np.ndarray, Image.Image],
    xy: Tuple[int, int],
    text: str,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = (255, 255, 255),
    font_size: int = 24,
    text_pos: str = 'left top'
) -> Image.Image:
    """
    Draw text on an image using PIL.

    Args:
        img: Input image (numpy BGR or PIL).
        xy: Position for text.
        text: Text to draw (supports \\n for newlines).
        background_color: Background color (RGB).
        text_color: Text color (RGB).
        font_size: Font size in pixels.
        text_pos: Anchor position ('left top', 'left down', 'right top').

    Returns:
        PIL Image with text drawn.
    """
    assert text_pos in ['left top', 'left down', 'right top']

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img[..., ::-1])  # BGR -> RGB

    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    xy = np.array(xy)
    texts = text.split('\n')
    if 'down' in text_pos:
        texts = texts[::-1]

    for t in texts:
        try:
            bbox = font.getbbox(t)
            w_text, h_text = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            w_text, h_text = font.getsize(t)

        if text_pos == 'left top':
            x_text, y_text = xy
        elif text_pos == 'left down':
            x_text, y_text = xy[0], xy[1] - h_text
        elif text_pos == 'right top':
            x_text, y_text = xy[0] - w_text, xy[1]
        else:
            x_text, y_text = xy

        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle(
            [x_text, y_text, x_text + w_text, y_text + h_text],
            radius=1.5,
            fill=background_color
        )
        draw.text((x_text, y_text), t, fill=text_color, font=font)
        xy[1] += h_text * (1 if 'top' in text_pos else -1)

    return img


def build_label2colors(classes: np.ndarray) -> dict:
    """
    Build a color mapping for class labels.

    Args:
        classes: Array of class indices.

    Returns:
        Dictionary mapping class index to RGB color tuple.
    """
    unique_cls = np.unique(classes.astype(int))
    return {
        c: tuple(np.random.RandomState(c).randint(0, 255, 3).tolist())
        for c in unique_cls
    }
