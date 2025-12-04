"""
Custom YOLO results class for Clash Royale detection.
Ported from KataCR/katacr/yolov8/custom_result.py

Handles detection results with additional 'bel' (belonging) attribute
to indicate friend (0) or enemy (1) units.
"""

import numpy as np
import torch
from pathlib import Path
from ultralytics.engine.results import Results, Boxes, ops, Annotator, deepcopy, LetterBox, colors, LOGGER, save_one_box

from ..processing.labels import idx2unit, idx2state


class CRBoxes(Boxes):
    """Custom Boxes class that includes belonging (bel) information."""

    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (7, 8), f"expected 7 or 8 values but got {n}"  # xyxy, track_id, conf, cls, bel
        assert isinstance(boxes, (torch.Tensor, np.ndarray))
        self.data = boxes
        self.orig_shape = orig_shape
        self.is_track = n == 8

    @property
    def id(self):
        """Return the track IDs of the boxes (if available)."""
        return self.data[:, -4] if self.is_track else None

    @property
    def cls(self):
        """Return class and belonging info."""
        return self.data[:, -2:]

    @property
    def conf(self):
        """Return confidence scores."""
        return self.data[:, -3]


class CRResults(Results):
    """
    Custom Results class for Clash Royale detection.

    Extends ultralytics Results with:
    - bel (belonging) attribute for friend/enemy classification
    - Custom visualization methods
    """

    def __init__(self, orig_img, path, names, boxes=None, logits_boxes=None,
                 masks=None, probs=None, keypoints=None, obb=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.orig_boxes = boxes
        self.logits_boxes = logits_boxes
        self.boxes = CRBoxes(boxes, self.orig_shape) if boxes is not None else None
        self.masks = None
        self.probs = None
        self.keypoints = None
        self.obb = None
        self.speed = {"preprocess": None, "inference": None, "postprocess": None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = "boxes", "masks", "probs", "keypoints", "obb"

    def update(self, boxes=None, masks=None, probs=None, obb=None):
        """Update the boxes, masks, and probs attributes."""
        if boxes is not None:
            self.boxes = CRBoxes(ops.clip_boxes(boxes, self.orig_shape), self.orig_shape)

    def get_data(self):
        """Get box data as numpy array: xyxy, (track_id), conf, cls, bel."""
        if not isinstance(self.boxes.data, np.ndarray):
            if self.boxes.data.device != 'cpu':
                self.boxes.data = self.boxes.data.cpu().numpy()
            else:
                self.boxes.data = self.boxes.data.numpy()
        return self.boxes.data

    def get_rgb(self):
        """Get original image in RGB format."""
        return self.orig_img[..., ::-1]  # BGR -> RGB

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
    ):
        """Plot detection results with belonging information."""
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes

        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil,
            example=names,
        )

        if pred_boxes is not None and show_boxes:
            for d in reversed(pred_boxes):
                c, bel = int(d.cls[0, 0]), int(d.cls[0, 1])
                conf_val = float(d.conf) if conf else None
                id_val = None if d.id is None else int(d.id.item())
                name = ("" if id_val is None else f"id:{id_val} ") + names[c] + str(bel)
                label = (f"{name} {conf_val:.2f}" if conf_val else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(box, label, color=colors(c, True), rotated=is_obb)

        if show:
            annotator.show(self.path)
        if save:
            annotator.save(filename)

        return annotator.result()

    def show_box(self, draw_center_point=False, verbose=False, use_overlay=True,
                 show_conf=False, save_path=None, fontsize=12, show_track=True):
        """Draw detection boxes with labels on the image."""
        from PIL import Image, ImageDraw, ImageFont

        img = self.get_rgb()
        box = self.get_data()
        img = img.copy()

        if isinstance(img, np.ndarray):
            if img.max() <= 1.0:
                img = (img * 255).astype('uint8')
            img = Image.fromarray(img.astype('uint8'))

        if use_overlay:
            overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))

        # Build color map for labels
        if len(box):
            unique_cls = set(box[:, -2].astype(int))
            label2color = {c: tuple(np.random.RandomState(c).randint(0, 255, 3).tolist())
                          for c in unique_cls}

        draw = ImageDraw.Draw(overlay if use_overlay else img)

        try:
            font = ImageFont.truetype("Arial.ttf", fontsize)
        except:
            font = ImageFont.load_default()

        for b in box:
            bel = int(b[-1])
            cls = int(b[-2])
            conf = float(b[-3])
            text = idx2unit.get(cls, f"cls{cls}")
            text += idx2state.get(bel, str(bel))
            if show_track and box.shape[1] == 8:
                text += ' ' + str(int(b[-4]))
            if show_conf:
                text += f' {conf:.2f}'

            x1, y1, x2, y2 = b[:4].astype(int)
            color = label2color.get(cls, (255, 0, 0))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, y1 - fontsize - 2), text, fill=color, font=font)

            if draw_center_point:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=color)

        if use_overlay:
            img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

        if verbose:
            img.show()
        if save_path is not None:
            img.save(str(save_path))

        return np.array(img)[..., ::-1]  # RGB -> BGR

    def verbose(self):
        """Return log string for detections."""
        log_string = ""
        boxes = self.boxes
        if len(self) == 0:
            return f"{log_string}(no detections), "
        if boxes:
            for c in boxes.cls[:, 0].unique():
                n = (boxes.cls[:, 0] == c).sum()
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string
