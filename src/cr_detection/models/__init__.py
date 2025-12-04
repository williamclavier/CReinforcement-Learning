"""
Detection models for Clash Royale game state detection.
"""

from .yolo_detector import ComboDetector, create_detector
from .custom_result import CRResults, CRBoxes

__all__ = ['ComboDetector', 'create_detector', 'CRResults', 'CRBoxes']
