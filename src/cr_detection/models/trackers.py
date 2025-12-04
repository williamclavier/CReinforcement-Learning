"""
Custom ByteTrack tracker for Clash Royale detection.
Ported from KataCR/katacr/yolov8/custom_trackers.py

Extends the standard ByteTrack with 'bel' (belonging) attribute
to track friend vs enemy units.
"""

from functools import partial
import numpy as np
import torch
import yaml
from pathlib import Path

from ultralytics.utils import IterableSimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker, STrack, matching, TrackState
from ultralytics.trackers.bot_sort import BOTSORT


def yaml_load(file_path):
    """Load a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def check_yaml(file_path):
    """Check and return the YAML file path."""
    return Path(file_path)


class CRSTrack(STrack):
    """Custom STrack with belonging (bel) attribute."""

    def __init__(self, xywh, score, cls, bel):
        """Initialize with bel (belonging) attribute."""
        super().__init__(xywh, score, cls)
        self.bel = bel

    def re_activate(self, new_track, frame_id, new_id=False):
        """Reactivate a previously lost track."""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.bel = new_track.bel
        self.angle = new_track.angle
        self.idx = new_track.idx

    @property
    def result(self):
        """Get current tracking results."""
        coords = self.xyxy if self.angle is None else self.xywha
        return coords.tolist() + [self.track_id, self.score, self.cls, self.bel, self.idx]

    def update(self, new_track, frame_id):
        """Update the state of a matched track."""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.convert_coords(new_tlwh)
        )
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.bel = new_track.bel
        self.angle = new_track.angle
        self.idx = new_track.idx


class CRBYTETracker(BYTETracker):
    """Custom ByteTracker that handles belonging (bel) attribute."""

    def update(self, results, img=None):
        """Update tracker with new detections."""
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        scores = results.conf
        bboxes = results.xywhr if hasattr(results, "xywhr") else results.xywh
        bboxes = np.concatenate([bboxes, np.arange(len(bboxes)).reshape(-1, 1)], axis=-1)
        cls = results.cls[:, 0]
        bel = results.cls[:, 1]

        remain_inds = scores > self.args.track_high_thresh
        inds_low = scores > self.args.track_low_thresh
        inds_high = scores < self.args.track_high_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = bboxes[inds_second]
        dets = bboxes[remain_inds]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]
        cls_keep = cls[remain_inds]
        cls_second = cls[inds_second]
        bel_keep = bel[remain_inds]
        bel_second = bel[inds_second]

        detections = self.init_track(dets, scores_keep, cls_keep, bel_keep, img)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        strack_pool = self.joint_stracks(tracked_stracks, self.lost_stracks)
        self.multi_predict(strack_pool)

        if hasattr(self, "gmc") and img is not None:
            warp = self.gmc.apply(img, dets)
            STrack.multi_gmc(strack_pool, warp)
            STrack.multi_gmc(unconfirmed, warp)

        dists = self.get_dists(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        detections_second = self.init_track(dets_second, scores_second, cls_second, bel_second, img)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if track.state != TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)

        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        for inew in u_detection:
            track = detections[inew]
            if track.score < self.args.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)

        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self.joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = self.sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = self.remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        if len(self.removed_stracks) > 1000:
            self.removed_stracks = self.removed_stracks[-999:]

        return np.asarray([x.result for x in self.tracked_stracks if x.is_activated], dtype=np.float32)

    def init_track(self, dets, scores, cls, bel, img=None):
        """Initialize tracks with belonging attribute."""
        return [CRSTrack(xyxy, s, c, b) for (xyxy, s, c, b) in zip(dets, scores, cls, bel)] if len(dets) else []


class CRBOTSORT(BOTSORT, CRBYTETracker):
    """Custom BOTSORT with belonging attribute."""
    pass


TRACKER_MAP = {"bytetrack": CRBYTETracker, "botsort": CRBOTSORT}


def cr_on_predict_start(detector, persist: bool = True) -> None:
    """Initialize tracker for the detector."""
    if detector.tracker is not None and persist:
        return

    tracker = check_yaml(detector.tracker_cfg_path)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))

    if cfg.tracker_type not in ["bytetrack", "botsort"]:
        raise AssertionError(f"Only 'bytetrack' and 'botsort' are supported, got '{cfg.tracker_type}'")

    detector.tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)


def cr_on_predict_postprocess_end(detector, persist: bool = True) -> None:
    """Update tracker with detection results."""
    det = detector.result.boxes.cpu().numpy()
    if len(det) == 0:
        return

    tracks = detector.tracker.update(det)
    if len(tracks) == 0:
        return

    update_args = dict(boxes=torch.as_tensor(tracks[:, :-1]))
    detector.result.update(**update_args)
