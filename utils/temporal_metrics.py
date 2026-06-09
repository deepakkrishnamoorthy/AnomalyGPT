"""Temporal interval metrics for frame-level anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TemporalInterval:
    """Inclusive 1-based frame interval."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return max(0, self.end - self.start + 1)


def merge_intervals(intervals: Iterable[TemporalInterval]) -> list[TemporalInterval]:
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    merged: list[TemporalInterval] = []
    for interval in ordered:
        if not merged or interval.start > merged[-1].end + 1:
            merged.append(interval)
            continue
        prev = merged[-1]
        merged[-1] = TemporalInterval(prev.start, max(prev.end, interval.end))
    return merged


def interval_union_length(intervals: Iterable[TemporalInterval]) -> int:
    return sum(interval.length for interval in merge_intervals(intervals))


def interval_intersection_length(
    predicted: Iterable[TemporalInterval],
    target: Iterable[TemporalInterval],
) -> int:
    left = merge_intervals(predicted)
    right = merge_intervals(target)
    i = j = total = 0
    while i < len(left) and j < len(right):
        start = max(left[i].start, right[j].start)
        end = min(left[i].end, right[j].end)
        if start <= end:
            total += end - start + 1
        if left[i].end < right[j].end:
            i += 1
        else:
            j += 1
    return total


def interval_iou(
    predicted: Iterable[TemporalInterval],
    target: Iterable[TemporalInterval],
) -> float:
    predicted_list = list(predicted)
    target_list = list(target)
    intersection = interval_intersection_length(predicted_list, target_list)
    union = interval_union_length([*predicted_list, *target_list])
    return intersection / union if union else 1.0


def intervals_to_frame_labels(
    intervals: Iterable[TemporalInterval],
    frame_count: int,
) -> list[int]:
    labels = [0] * frame_count
    for interval in intervals:
        start = max(1, interval.start)
        end = min(frame_count, interval.end)
        for frame_id in range(start, end + 1):
            labels[frame_id - 1] = 1
    return labels


def threshold_scores_to_intervals(
    scores: Sequence[float],
    threshold: float,
    *,
    first_frame_id: int = 1,
) -> list[TemporalInterval]:
    intervals: list[TemporalInterval] = []
    start: int | None = None
    for offset, score in enumerate(scores):
        frame_id = first_frame_id + offset
        if score >= threshold and start is None:
            start = frame_id
        elif score < threshold and start is not None:
            intervals.append(TemporalInterval(start=start, end=frame_id - 1))
            start = None
    if start is not None:
        intervals.append(TemporalInterval(start=start, end=first_frame_id + len(scores) - 1))
    return intervals
