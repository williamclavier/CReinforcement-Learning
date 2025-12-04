"""
State builder for Clash Royale game state detection.
Simplified version ported from KataCR/katacr/policy/perceptron/state_builder.py

Builds a unified game state from detection results:
- Unit positions and types
- Tower health
- Game time
"""

import numpy as np
import scipy.spatial
from collections import Counter, defaultdict
from queue import Queue
from typing import Dict, List, Optional, Set, Tuple

from ..models.custom_result import CRResults
from ..ocr.paddle_ocr import OCR
from ..processing.labels import idx2unit, unit2idx
from ..utils.detection_utils import (
    pixel2cell, cell2pixel, xyxy2center, extract_img, xyxy2sub, background_size
)

# Constants
BAR_CENTER2BODY_DELTA_Y = 40
DIS_BAR_AND_BAR_LEVEL_THRE = 15
DIS_BAR_AND_BODY_THRE = 35

# Tower unit names
TOWER_UNITS = {'king-tower', 'queen-tower', 'cannoneer-tower', 'dagger-duchess-tower'}
EXCEPT_KING_TOWER = {'queen-tower', 'cannoneer-tower', 'dagger-duchess-tower'}

# Spell and object units (no health bars)
SPELL_UNITS = {
    'arrows', 'clone', 'earthquake', 'fireball', 'freeze', 'giant-snowball',
    'goblin-barrel', 'graveyard', 'lightning', 'poison', 'rage', 'rocket',
    'skeleton-king-skill', 'tesla-evolution-shock', 'tornado', 'zap',
}
OBJECT_UNITS = {'axe', 'dirt', 'goblin-ball', 'bomb'}


class UnitInfo:
    """Information about a detected unit."""

    def __init__(
        self,
        xy: Optional[np.ndarray] = None,
        cls: Optional[int] = None,
        bel: Optional[int] = None,
        conf: float = 0.0,
        track_id: Optional[int] = None
    ):
        self.xy = xy  # Position in cell coordinates
        self.cls = cls  # Class index
        self.bel = bel  # Belonging (0=friend, 1=enemy)
        self.conf = conf  # Confidence
        self.track_id = track_id

    @property
    def class_name(self) -> str:
        """Get the unit class name."""
        return idx2unit.get(self.cls, f'unknown_{self.cls}') if self.cls is not None else 'unknown'

    @property
    def is_friendly(self) -> bool:
        """Check if unit belongs to player."""
        return self.bel == 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'xy': self.xy.tolist() if self.xy is not None else None,
            'cls': self.cls,
            'class_name': self.class_name,
            'bel': self.bel,
            'conf': self.conf,
            'track_id': self.track_id,
        }


class GameState:
    """
    Complete game state at a single frame.

    Attributes:
        time: Game time in seconds (0-300).
        units: List of detected units.
        towers: Dictionary of tower states.
    """

    def __init__(self):
        self.time: float = 0
        self.units: List[UnitInfo] = []
        self.towers: Dict[str, dict] = {}
        self.frame_count: int = 0

    def get_friendly_units(self) -> List[UnitInfo]:
        """Get all friendly units."""
        return [u for u in self.units if u.bel == 0]

    def get_enemy_units(self) -> List[UnitInfo]:
        """Get all enemy units."""
        return [u for u in self.units if u.bel == 1]

    def get_units_by_type(self, cls_name: str) -> List[UnitInfo]:
        """Get units of a specific type."""
        cls_idx = unit2idx.get(cls_name)
        if cls_idx is None:
            return []
        return [u for u in self.units if u.cls == cls_idx]

    def to_dict(self) -> dict:
        """Convert state to dictionary."""
        return {
            'time': self.time,
            'frame_count': self.frame_count,
            'units': [u.to_dict() for u in self.units],
            'towers': self.towers,
            'friendly_count': len(self.get_friendly_units()),
            'enemy_count': len(self.get_enemy_units()),
        }


class StateBuilder:
    """
    Builds game state from detection results.

    Combines YOLO detections and OCR results into a unified game state.
    """

    def __init__(self, persist: int = 3, ocr: Optional[OCR] = None):
        """
        Initialize state builder.

        Args:
            persist: Memory persistence time in seconds.
            ocr: OCR instance for text detection.
        """
        self.persist = persist
        self.ocr = OCR(lang='en') if ocr is None else ocr
        self.reset()

    def reset(self):
        """Reset all state."""
        self.time = 0
        self.frame_count = 0
        self.bel_memory: Dict[int, Counter] = defaultdict(Counter)
        self.cls_memory: Dict[int, Counter] = defaultdict(Counter)
        self._current_state = GameState()

    def update(self, arena_result: CRResults, time: float = 0) -> GameState:
        """
        Update state from detection results.

        Args:
            arena_result: CRResults from YOLOv8 detection.
            time: Game time from OCR.

        Returns:
            Updated GameState.
        """
        self.time = time if not np.isinf(time) else self.time
        self.frame_count += 1

        box = arena_result.get_data()
        img = arena_result.get_rgb()

        # Build new state
        state = GameState()
        state.time = self.time
        state.frame_count = self.frame_count

        # Update belonging memory and process detections
        has_track = box.shape[1] == 8 if len(box) > 0 else False

        for b in box:
            if has_track:
                x1, y1, x2, y2, track_id, conf, cls, bel = b
                track_id = int(track_id)
            else:
                x1, y1, x2, y2, conf, cls, bel = b[:7]
                track_id = None

            cls = int(cls)
            bel = int(bel)
            cls_name = idx2unit.get(cls, '')

            # Skip UI elements
            if cls_name in {'bar', 'bar-level', 'clock', 'emote', 'text', 'elixir', 'selected'}:
                continue

            # Skip bar elements (handled separately)
            if 'bar' in cls_name and cls_name not in TOWER_UNITS:
                continue

            # Update memory for tracking consistency
            if track_id is not None:
                self.bel_memory[track_id].update({bel: 1})
                self.cls_memory[track_id].update({cls: 1})

                # Use most common belonging
                most_bel = self.bel_memory[track_id].most_common(1)[0][0]
                bel = most_bel

            # Create unit info
            center = xyxy2center([x1, y1, x2, y2])
            xy_cell = pixel2cell(center)

            unit = UnitInfo(
                xy=xy_cell,
                cls=cls,
                bel=bel,
                conf=float(conf),
                track_id=track_id
            )
            state.units.append(unit)

            # Track tower states
            if cls_name in TOWER_UNITS:
                tower_key = f"{cls_name}_{bel}"
                state.towers[tower_key] = {
                    'type': cls_name,
                    'bel': bel,
                    'position': center.tolist(),
                    'alive': True,
                }

        self._current_state = state
        return state

    def get_state(self) -> GameState:
        """Get the current game state."""
        return self._current_state

    def get_unit_count(self) -> Tuple[int, int]:
        """
        Get count of friendly and enemy units.

        Returns:
            Tuple of (friendly_count, enemy_count).
        """
        state = self._current_state
        return (
            len(state.get_friendly_units()),
            len(state.get_enemy_units())
        )

    def get_units_near(
        self,
        xy: Tuple[float, float],
        radius: float = 3.0,
        bel: Optional[int] = None
    ) -> List[UnitInfo]:
        """
        Get units within a radius of a position.

        Args:
            xy: Position in cell coordinates.
            radius: Search radius in cells.
            bel: Filter by belonging (0=friendly, 1=enemy).

        Returns:
            List of nearby units.
        """
        state = self._current_state
        xy = np.array(xy)
        nearby = []

        for unit in state.units:
            if unit.xy is None:
                continue
            if bel is not None and unit.bel != bel:
                continue

            dist = np.linalg.norm(unit.xy - xy)
            if dist <= radius:
                nearby.append(unit)

        return nearby
