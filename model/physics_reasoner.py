"""Trajectory-based physics features for video anomaly reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TrackPoint:
    frame: int
    x: float
    y: float


@dataclass(frozen=True)
class PhysicsFeatures:
    track_id: int
    velocity: float
    acceleration: float
    direction: float
    trajectory_score: float
    flow_deviation: float


def _velocities(points: Sequence[TrackPoint], fps: float) -> list[tuple[float, float]]:
    velocities: list[tuple[float, float]] = []
    for prev, curr in zip(points, points[1:]):
        dt = max((curr.frame - prev.frame) / fps, 1e-6)
        velocities.append(((curr.x - prev.x) / dt, (curr.y - prev.y) / dt))
    return velocities


def _mean_direction(velocities: Iterable[tuple[float, float]]) -> float:
    vectors = list(velocities)
    if not vectors:
        return 0.0
    return degrees(atan2(mean(vy for _, vy in vectors), mean(vx for vx, _ in vectors)))


def direction_deviation(direction: float, crowd_direction: float) -> float:
    """Return the smallest angular difference in degrees."""
    diff = abs((direction - crowd_direction + 180.0) % 360.0 - 180.0)
    return diff


def extract_track_physics(
    track_id: int,
    points: Sequence[TrackPoint],
    fps: float,
    crowd_direction: float = 0.0,
) -> PhysicsFeatures:
    """Compute compact physics features from one tracked trajectory."""
    velocities = _velocities(points, fps)
    speeds = [hypot(vx, vy) for vx, vy in velocities]
    velocity = mean(speeds) if speeds else 0.0

    accelerations = [
        abs(curr - prev) * fps
        for prev, curr in zip(speeds, speeds[1:])
    ]
    acceleration = mean(accelerations) if accelerations else 0.0
    direction = _mean_direction(velocities)
    flow_deviation = direction_deviation(direction, crowd_direction)

    # Placeholder score until normal trajectory models are trained.
    trajectory_score = min(flow_deviation / 180.0, 1.0)

    return PhysicsFeatures(
        track_id=track_id,
        velocity=velocity,
        acceleration=acceleration,
        direction=direction,
        trajectory_score=trajectory_score,
        flow_deviation=flow_deviation,
    )
