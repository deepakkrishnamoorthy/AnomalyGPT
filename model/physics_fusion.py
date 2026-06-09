"""Fusion boundary for visual, language, and physics features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PhysicsFusionConfig:
    visual_dim: int
    language_dim: int
    physics_dim: int
    hidden_dim: int


class PhysicsFusion:
    """Framework-neutral placeholder for the future PyTorch fusion module."""

    def __init__(self, config: PhysicsFusionConfig) -> None:
        self.config = config

    def __call__(
        self,
        visual_features: Any,
        language_features: Any,
        physics_features: Any,
    ) -> dict[str, Any]:
        return {
            "visual_features": visual_features,
            "language_features": language_features,
            "physics_features": physics_features,
            "config": self.config,
        }
