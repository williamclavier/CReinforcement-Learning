"""
Custom predictor for Clash Royale detection.
Ported from KataCR/katacr/yolov8/custom_predict.py

This predictor applies custom NMS that outputs 7 columns including 'belong'.
"""

from ultralytics.engine.predictor import BasePredictor
from ultralytics.utils import ops

from .custom_nms import non_max_suppression
from .custom_result import CRResults


class CRDetectionPredictor(BasePredictor):
    """
    Custom predictor for CR detection models.

    Applies custom NMS that handles the 'belong' attribute (friend vs enemy).
    """

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions and returns a list of CRResults objects."""
        # Don't pass nc - let NMS compute it from prediction shape
        # The model outputs (4 + nc) where nc includes both classes AND 'bel'
        # If we pass model.nc, it only has the actual class count without 'bel'
        preds = non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            agnostic=self.args.agnostic_nms,
            max_det=self.args.max_det,
            classes=self.args.classes,
            # nc is intentionally omitted - computed from prediction.shape[1] - 4
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            if len(pred) > 0:
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            img_path = self.batch[0][i]
            results.append(CRResults(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
