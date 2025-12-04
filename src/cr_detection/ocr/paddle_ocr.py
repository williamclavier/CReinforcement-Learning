"""
PaddleOCR wrapper for Clash Royale text recognition.
Ported from KataCR/katacr/ocr_text/paddle_ocr.py

Used for:
- Game time detection
- Tower health detection
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import Optional, Union, List, Tuple


class OCR:
    """
    OCR wrapper for Clash Royale game state detection.

    Uses PaddleOCR for text recognition of:
    - Game timer (top right)
    - Tower health values
    """

    # Episode state flags
    START_EPISODE_FLAG = 0
    END_EPISODE_FLAG = 1

    def __init__(
        self,
        use_angle_cls: bool = False,
        onnx: bool = False,
        tensorrt: bool = False,
        use_gpu: bool = True,
        lang: str = 'en'
    ):
        """
        Initialize OCR.

        Args:
            use_angle_cls: Whether to use angle classification.
            onnx: Whether to use ONNX models.
            tensorrt: Whether to use TensorRT.
            use_gpu: Whether to use GPU acceleration.
            lang: Language for OCR ('en', 'ch', etc.).
        """
        self.use_angle_cls = use_angle_cls
        kwargs = dict(use_onnx=onnx, use_tensorrt=tensorrt, use_gpu=use_gpu)
        self.ocr = PaddleOCR(use_angle_cls=use_angle_cls, show_log=False, lang=lang, **kwargs)

    def __call__(
        self,
        x: Union[np.ndarray, str, List],
        det: bool = True,
        rec: bool = True,
        cls: bool = False,
        bin: bool = False,
        pil: bool = True,
        gray: bool = False
    ) -> List:
        """
        Run OCR on image(s).

        Args:
            x: Image array, path, or list of images.
            det: Use text position detection.
            rec: Use text recognition.
            cls: Recognize 180-degree rotated text.
            bin: Binarize image to grayscale.
            pil: Image is RGB format.
            gray: Image has one color channel.

        Returns:
            List of OCR results.
        """
        if not pil and not gray:
            if isinstance(x, list):
                for i in range(len(x)):
                    if isinstance(x[i], np.ndarray):
                        x[i] = x[i][..., ::-1]  # BGR -> RGB
            elif isinstance(x, np.ndarray):
                x = x[..., ::-1]

        cls = cls & self.use_angle_cls
        result = self.ocr.ocr(x, det=det, rec=rec, cls=cls, bin=bin)
        return result

    def process_part1(self, img_time: np.ndarray, pil: bool = False, show: bool = False) -> float:
        """
        Process time display area to extract game time.

        Args:
            img_time: Image of the time area (part1).
            pil: Image is RGB format.
            show: Show debug visualization.

        Returns:
            Game time in seconds (0-300), or np.inf if not detected.
        """
        results = self(img_time, pil=pil)[0]

        if show:
            print("OCR results:", results)
            cv2.imshow('time', img_time)
            cv2.waitKey(1)

        if results is None:
            return np.inf

        stage = m = s = None
        for info in results:
            det, rec = info
            rec = rec[0].lower()
            if 'left' in rec:
                stage = 0  # Regular time
            if 'over' in rec:
                stage = 1  # Overtime
            if (':' in rec) or ('：' in rec):
                m, s = rec.split(':' if ':' in rec else '：')
                try:
                    m = int(m.strip())
                    s = int(s.strip())
                except ValueError:
                    m = s = None

        if stage is None or m is None or s is None:
            return np.inf

        t = m * 60 + s
        if stage == 0:
            return 180 - t  # Regular time: 3 minutes countdown
        return 180 + 120 - t  # Overtime: 2 more minutes

    def process_tower_hp(
        self,
        img: np.ndarray,
        target_size: Tuple[int, int],
        pil: bool = False,
        conf_threshold: float = 0.9
    ) -> int:
        """
        Process tower health bar image to extract HP value.

        Args:
            img: Image of the tower health bar area.
            target_size: Target size to resize image to.
            pil: Image is RGB format.
            conf_threshold: Minimum confidence for OCR result.

        Returns:
            Tower HP as integer, or -1 if not detected.
        """
        # Resize to target size
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)

        # Try both normal and flipped versions
        imgs = [img_resized, img_resized[..., ::-1]]
        results = self(imgs, det=False, pil=pil)[0]

        nums = []
        for rec, conf in results:
            rec = rec.lower()
            num = ''.join([c for c in rec.strip() if c.isdigit()])
            if conf < conf_threshold or len(num) == 0:
                num = -1
            else:
                num = int(num)
            nums.append(num)

        if len(nums) >= 2 and np.mean(nums) == nums[0]:
            return nums[0]
        return -1

    def process_center_texts(self, img: np.ndarray, pil: bool = False) -> Optional[int]:
        """
        Check for game state text in center of screen.

        Args:
            img: Full game image.
            pil: Image is RGB format.

        Returns:
            START_EPISODE_FLAG (0), END_EPISODE_FLAG (1), or None.
        """
        h, w = img.shape[:2]
        center_h = int(h * 0.43)
        target_h = int(h * 0.23)
        x0, y0, x1, y1 = [0, center_h - target_h // 2, w, center_h + target_h // 2]
        center_img = img[y0:y1, x0:x1]

        ratio = 300 / target_h
        center_img = cv2.resize(center_img, (int(w * ratio), int(target_h * ratio)))

        results = self(center_img, pil=pil)[0]

        if results is None:
            return None

        text_features_episode_end = ['match', 'over', 'break']

        recs = [info[1][0] for info in results]
        for text in recs:
            for flag, texts in zip(
                (self.START_EPISODE_FLAG, self.END_EPISODE_FLAG),
                (['fight'], text_features_episode_end)
            ):
                for target in texts:
                    if target.lower() in text.lower():
                        return flag
        return None


def create_ocr(use_gpu: bool = True, lang: str = 'en') -> OCR:
    """
    Factory function to create OCR instance.

    Args:
        use_gpu: Whether to use GPU acceleration.
        lang: Language for OCR.

    Returns:
        Configured OCR instance.
    """
    return OCR(use_gpu=use_gpu, lang=lang)
