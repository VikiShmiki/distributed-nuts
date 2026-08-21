"""Neal's funnel benchmark target."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from abnuts.models.base import ModelMetadata

LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class FunnelModel:
    """Neal's funnel with one scale coordinate and ``dimension - 1`` latent axes."""

    dimension: int
    scale_std: float = 3.0

    name: str = "funnel"

    def __post_init__(self) -> None:
        """Validate model dimensions."""
        if self.dimension < 2:
            raise ValueError(
                "Neal's funnel requires dimension >= 2 "
                "(one scale coordinate plus at least one latent coordinate)."
            )
        if self.scale_std <= 0.0:
            raise ValueError(f"scale_std must be positive, got {self.scale_std!r}")

    @property
    def metadata(self) -> ModelMetadata:
        """Return serializable metadata for result manifests."""
        return ModelMetadata(
            name=self.name,
            dimension=self.dimension,
            event_shape=(self.dimension,),
            description=(
                "Neal's funnel with y ~ Normal(0, scale_std) and "
                "x_i | y ~ Normal(0, exp(y / 2))."
            ),
        )

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate deterministic vectorized initial positions for many chains."""
        if num_chains <= 0:
            raise ValueError(f"num_chains must be positive, got {num_chains!r}")

        jitter_scale = float((config or {}).get("initial_jitter_scale", 0.1))
        if jitter_scale <= 0.0:
            raise ValueError(f"initial_jitter_scale must be positive, got {jitter_scale!r}")

        rng = random.Random(int(key))
        return [
            [rng.gauss(0.0, jitter_scale) for _ in range(self.dimension)]
            for _ in range(num_chains)
        ]

    def log_prob(self, position: Sequence[float] | Any, data: Any | None = None) -> Any:
        """Evaluate Neal's funnel log density, including normalizing constants."""
        del data
        if hasattr(position, "shape"):
            position_array = jnp.asarray(position)
            if position_array.shape != (self.dimension,):
                raise ValueError(
                    f"Expected position with shape ({self.dimension},), "
                    f"got {position_array.shape}"
                )

            y = position_array[0]
            xs = position_array[1:]

            scale_log_prob = -0.5 * (y / self.scale_std) ** 2
            scale_log_prob -= math.log(self.scale_std) + 0.5 * LOG_TWO_PI

            inv_variance = jnp.exp(-y)
            conditional_log_prob = -0.5 * jnp.sum(xs * xs * inv_variance + y + LOG_TWO_PI)
            return scale_log_prob + conditional_log_prob

        if len(position) != self.dimension:
            raise ValueError(
                f"Expected position with dimension {self.dimension}, got {len(position)}"
            )

        y = float(position[0])
        xs = [float(value) for value in position[1:]]

        scale_log_prob = -0.5 * (y / self.scale_std) ** 2
        scale_log_prob -= math.log(self.scale_std) + 0.5 * LOG_TWO_PI

        conditional_log_prob = 0.0
        inv_variance = math.exp(-y)
        for value in xs:
            conditional_log_prob -= 0.5 * (value * value * inv_variance + y + LOG_TWO_PI)

        return scale_log_prob + conditional_log_prob
