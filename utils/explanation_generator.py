"""Convert physics measurements into concise anomaly explanations."""

from __future__ import annotations

from typing import Mapping


def generate_explanation(physics: Mapping[str, float]) -> str:
    velocity_ratio = physics.get("velocity_ratio", 0.0)
    direction_deviation = physics.get("direction_deviation", 0.0)

    reasons: list[str] = []
    if velocity_ratio >= 3.0:
        reasons.append("moving significantly faster than nearby pedestrians")
    if direction_deviation >= 135.0:
        reasons.append("moving opposite to the dominant crowd direction")
    elif direction_deviation >= 75.0:
        reasons.append("deviating from the dominant crowd flow")

    if not reasons:
        return "The behavior is unusual relative to the learned normal scene patterns."

    return "The object is " + " and ".join(reasons) + "."
