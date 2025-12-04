"""
YOLOv8 Combo Detector for Clash Royale unit detection.
Ported from KataCR/katacr/yolov8/combo_detect.py

Uses two YOLOv8 models for multi-scale detection with ByteTrack for tracking.
"""

from pathlib import Path
from typing import Optional, List, Union
import sys
import types

import cv2
import numpy as np
import torch
import torchvision
from ultralytics import YOLO
from ultralytics.engine.model import Model

from ..processing.labels import unit2idx, idx2unit
from ..processing.constants import MODELS_DIR, DETECTION_CONF_THRESHOLD, IOU_THRESHOLD
from .custom_result import CRResults
from .custom_predict import CRDetectionPredictor
from .trackers import cr_on_predict_start, cr_on_predict_postprocess_end


# Default model paths
DEFAULT_MODEL_PATHS = [
    MODELS_DIR / 'detector1_v0.7.13.pt',
    MODELS_DIR / 'detector2_v0.7.13.pt',
]


def _patch_katacr_modules():
    """
    Patch sys.modules to allow loading models trained with KataCR.

    The models were saved with references to 'katacr' module which doesn't exist
    in this project. We create stub modules to allow torch.load to work.
    """
    # Create stub katacr module hierarchy
    if 'katacr' not in sys.modules:
        katacr = types.ModuleType('katacr')
        sys.modules['katacr'] = katacr

        # Create submodules as needed
        katacr_yolov8 = types.ModuleType('katacr.yolov8')
        sys.modules['katacr.yolov8'] = katacr_yolov8
        katacr.yolov8 = katacr_yolov8

        # Create custom_model stub (the main thing needed for loading)
        katacr_yolov8_custom_model = types.ModuleType('katacr.yolov8.custom_model')
        sys.modules['katacr.yolov8.custom_model'] = katacr_yolov8_custom_model
        katacr_yolov8.custom_model = katacr_yolov8_custom_model

        # Import the actual detection model from ultralytics
        from ultralytics.nn.tasks import DetectionModel, v8DetectionLoss
        from ultralytics.utils.tal import TaskAlignedAssigner

        # Create CRDetectionModel as an alias
        class CRDetectionModel(DetectionModel):
            """Stub for KataCR's CRDetectionModel - just uses standard DetectionModel."""
            def init_criterion(self):
                return CRDetectionLoss(self)

        # Create CRDetectionLoss stub
        class CRDetectionLoss(v8DetectionLoss):
            """Stub for KataCR's CRDetectionLoss."""
            pass

        # Create CRTaskAlignedAssigner stub
        class CRTaskAlignedAssigner(TaskAlignedAssigner):
            """Stub for KataCR's CRTaskAlignedAssigner."""
            pass

        katacr_yolov8_custom_model.CRDetectionModel = CRDetectionModel
        katacr_yolov8_custom_model.CRDetectionLoss = CRDetectionLoss
        katacr_yolov8_custom_model.CRTaskAlignedAssigner = CRTaskAlignedAssigner


# Apply patch when module is loaded
_patch_katacr_modules()


class YOLO_CR(Model):
    """
    Custom YOLO class for Clash Royale detection.

    Extends ultralytics Model to use our custom predictor that handles
    the 'belong' attribute (friend vs enemy classification).

    This is the key difference from standard YOLO - the task_map ensures
    our CRDetectionPredictor is used instead of the default predictor.
    """

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        # Import here to get the patched CRDetectionModel
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator

        return {
            "detect": {
                "model": DetectionModel,
                "trainer": DetectionTrainer,
                "validator": DetectionValidator,
                "predictor": CRDetectionPredictor,  # Our custom predictor!
            },
        }

    def track(self, source=None, stream=False, persist=False, **kwargs) -> list:
        """Track objects with ByteTrack."""
        if not hasattr(self.predictor, "trackers"):
            from .trackers import register_tracker
            register_tracker(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack needs low confidence
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)


class ComboDetector:
    """
    Multi-model YOLOv8 detector with ByteTrack tracking for Clash Royale.

    Combines detections from multiple YOLO models and applies NMS
    and optional object tracking.
    """

    def __init__(
        self,
        model_paths: Optional[List[Union[str, Path]]] = None,
        show_conf: bool = True,
        conf: float = DETECTION_CONF_THRESHOLD,
        iou_thre: float = IOU_THRESHOLD,
        tracker: Optional[str] = 'bytetrack',
        device: Optional[str] = None
    ):
        """
        Initialize the combo detector.

        Args:
            model_paths: List of paths to YOLO model files. Uses defaults if None.
            show_conf: Whether to show confidence in visualizations.
            conf: Confidence threshold for detections.
            iou_thre: IOU threshold for NMS.
            tracker: Tracker type ('bytetrack', 'botsort', or None for no tracking).
            device: Device to run on ('mps', 'cuda', 'cpu', or None for auto-detect).
        """
        if model_paths is None:
            model_paths = DEFAULT_MODEL_PATHS

        # Auto-detect best device
        if device is None:
            if torch.backends.mps.is_available():
                device = 'mps'  # Apple Silicon GPU
            elif torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        self.device = device

        # Load models using YOLO_CR which registers our custom predictor via task_map
        self.models = [YOLO_CR(str(p)) for p in model_paths]

        # Move models to device
        for model in self.models:
            model.to(device)

        # Warm up MPS by running a dummy inference (avoids first-run lag)
        if device == 'mps':
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for model in self.models:
                model.predict(dummy, verbose=False)

        self.show_conf = show_conf
        self.conf = conf
        self.iou_thre = iou_thre
        self.tracker = None
        self.result: Optional[CRResults] = None

        if tracker == 'bytetrack':
            self.conf = 0.1  # ByteTrack needs low conf for tracking
            self.tracker_cfg_path = str(MODELS_DIR / 'bytetrack.yaml')
            cr_on_predict_start(self, persist=True)

    def infer(self, x: np.ndarray, pil: bool = False) -> CRResults:
        """
        Run inference on a single image.

        Follows the same approach as KataCR/katacr/yolov8/combo_detect.py

        Args:
            x: Image array with shape (H, W, 3).
            pil: If True, input is RGB format. If False, input is BGR.

        Returns:
            CRResults object with detection data.
        """
        if pil:
            x = x[..., ::-1]  # RGB -> BGR

        # Run all models with custom predictor
        results = [m.predict(x, verbose=False, conf=self.conf)[0] for m in self.models]

        # Combine predictions from all models
        preds = []
        for p in results:
            boxes = p.orig_boxes.clone()
            for i in range(len(boxes)):
                # Map class name to our unified index
                cls_name = p.names[int(boxes[i, 5])]
                if cls_name in unit2idx:
                    boxes[i, 5] = unit2idx[cls_name]
                    preds.append(boxes[i])

        if not preds:
            preds = torch.zeros(0, 7)
        else:
            preds = torch.cat(preds, 0).reshape(-1, 7)

        # Apply NMS across combined predictions
        if len(preds) > 0:
            i = torchvision.ops.nms(preds[:, :4], preds[:, 4], iou_threshold=self.iou_thre)
            preds = preds[i]

        # Create result
        self.result = CRResults(x, path="", names=idx2unit, boxes=preds)

        # Apply tracking if enabled
        if self.tracker is not None:
            cr_on_predict_postprocess_end(self, persist=True)

        # Filter out detections in UI areas
        data = self.result.get_data()
        if len(data) > 0:
            mask = ~(((data[:, 0] > 390) & (data[:, 3] < 120)) |
                     ((data[:, 2] < 280) & (data[:, 3] < 80)))
            self.result.boxes.data = data[mask]

        return self.result

    def detect(self, image: np.ndarray, rgb: bool = True) -> CRResults:
        """
        Convenience method for detection.

        Args:
            image: Input image as numpy array.
            rgb: If True, image is RGB. If False, image is BGR.

        Returns:
            CRResults object with detection data.
        """
        return self.infer(image, pil=rgb)

    def reset_tracker(self):
        """Reset the object tracker state."""
        if self.tracker is not None:
            cr_on_predict_start(self, persist=False)

    def get_detections(self) -> dict:
        """
        Get detections as a dictionary.

        Returns:
            Dictionary with keys:
            - 'boxes': List of [x1, y1, x2, y2] coordinates
            - 'track_ids': List of tracking IDs (if tracking enabled)
            - 'confidences': List of confidence scores
            - 'classes': List of class indices
            - 'class_names': List of class names
            - 'belongs': List of belonging values (0=friend, 1=enemy)
        """
        if self.result is None:
            return {'boxes': [], 'track_ids': [], 'confidences': [],
                   'classes': [], 'class_names': [], 'belongs': []}

        data = self.result.get_data()
        has_track = data.shape[1] == 8

        return {
            'boxes': data[:, :4].tolist(),
            'track_ids': data[:, -4].astype(int).tolist() if has_track else [],
            'confidences': data[:, -3].tolist(),
            'classes': data[:, -2].astype(int).tolist(),
            'class_names': [idx2unit.get(int(c), f'unknown_{c}') for c in data[:, -2]],
            'belongs': data[:, -1].astype(int).tolist(),
        }


def create_detector(
    model_paths: Optional[List[str]] = None,
    use_tracking: bool = True
) -> ComboDetector:
    """
    Factory function to create a detector.

    Args:
        model_paths: Optional list of model file paths.
        use_tracking: Whether to enable object tracking.

    Returns:
        Configured ComboDetector instance.
    """
    tracker = 'bytetrack' if use_tracking else None
    return ComboDetector(model_paths=model_paths, tracker=tracker)
