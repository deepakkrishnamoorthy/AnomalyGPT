"""Tracker adapter boundary for ByteTrack or BoT-SORT integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Detection:
    frame: int
    bbox: tuple[float, float, float, float]
    score: float
    category: str = "object"


@dataclass(frozen=True)
class Track:
    track_id: int
    category: str
    boxes: Sequence[Detection]


class TrackerAdapter:
    """Small interface to keep tracker choice isolated from dataset building."""

    def track(self, detections: Sequence[Detection]) -> list[Track]:
        raise NotImplementedError("Connect ByteTrack or BoT-SORT here.")
