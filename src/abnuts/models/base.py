"""Shared benchmark model interfaces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Position = Sequence[float]
PositionBatch = Sequence[Position]


@dataclass(frozen=True)
class ModelMetadata:
    """Small serializable description of a benchmark target."""

    name: str
    dimension: int
    event_shape: tuple[int, ...]
    description: str
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-friendly model metadata."""
        metadata = {
            "name": self.name,
            "dimension": self.dimension,
            "event_shape": list(self.event_shape),
            "description": self.description,
        }
        metadata.update(self.extra)
        return metadata


class BenchmarkModel(Protocol):
    """Protocol shared by benchmark targets used by experiments."""

    name: str
    dimension: int

    @property
    def metadata(self) -> ModelMetadata:
        """Return serializable model metadata."""

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate a deterministic batch of initial positions."""

    def log_prob(self, position: Any, data: Any | None = None) -> Any:
        """Evaluate the unnormalized log probability for one position."""


def vectorized_log_prob(
    model: BenchmarkModel,
    positions: PositionBatch,
    data: Any | None = None,
) -> list[float]:
    """Evaluate a model's log probability for a batch of chain positions."""
    return [model.log_prob(position, data=data) for position in positions]
