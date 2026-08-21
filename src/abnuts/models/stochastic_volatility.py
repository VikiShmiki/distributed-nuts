"""Synthetic stochastic volatility benchmark target."""

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
class StochasticVolatilityData:
    """Deterministic observations and latent states for a volatility time series."""

    observations: tuple[float, ...]
    true_log_volatility: tuple[float, ...]
    persistence: float
    innovation_scale: float

    @property
    def time_length(self) -> int:
        """Return the number of time points in the series."""
        return len(self.observations)


@dataclass(frozen=True)
class StochasticVolatilityModel:
    """Latent AR(1) log-volatility model with fixed dynamics parameters."""

    dimension: int
    time_length: int | None = None
    persistence: float = 0.95
    innovation_scale: float = 0.2
    data_seed: int = 29
    synthetic_data: StochasticVolatilityData | None = None
    name: str = "stochastic_volatility"

    def __post_init__(self) -> None:
        """Validate dimensions and generate deterministic synthetic observations."""
        time_length = self.dimension if self.time_length is None else int(self.time_length)
        if time_length <= 0:
            raise ValueError(f"time_length must be positive, got {time_length!r}")
        if self.dimension != time_length:
            raise ValueError(
                f"{self.name} requires dimension equal to time_length "
                f"({time_length}); got {self.dimension!r}"
            )
        if not -1.0 < self.persistence < 1.0:
            raise ValueError(
                f"persistence must be strictly between -1 and 1, got {self.persistence!r}"
            )
        if self.innovation_scale <= 0.0:
            raise ValueError(
                f"innovation_scale must be positive, got {self.innovation_scale!r}"
            )
        object.__setattr__(self, "time_length", time_length)

        data = self.synthetic_data
        if data is None:
            data = generate_synthetic_stochastic_volatility_data(
                time_length=time_length,
                persistence=self.persistence,
                innovation_scale=self.innovation_scale,
                seed=self.data_seed,
            )
            object.__setattr__(self, "synthetic_data", data)
        self._validate_data(data)

    @property
    def metadata(self) -> ModelMetadata:
        """Return serializable metadata for result manifests."""
        data = self._require_data()
        return ModelMetadata(
            name=self.name,
            dimension=self.dimension,
            event_shape=(self.dimension,),
            description=(
                "Synthetic stochastic volatility benchmark with latent AR(1) "
                "log-volatility states and fixed dynamics parameters."
            ),
            extra={
                "model_family": "stochastic_volatility",
                "time_length": self.time_length,
                "num_observations": data.time_length,
                "persistence": self.persistence,
                "innovation_scale": self.innovation_scale,
                "data_seed": self.data_seed,
            },
        )

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate deterministic initial latent log-volatility sequences."""
        if num_chains <= 0:
            raise ValueError(f"num_chains must be positive, got {num_chains!r}")

        jitter_scale = float((config or {}).get("initial_jitter_scale", 0.05))
        if jitter_scale <= 0.0:
            raise ValueError(f"initial_jitter_scale must be positive, got {jitter_scale!r}")

        rng = random.Random(int(key))
        data = self._require_data()
        positions: list[list[float]] = []
        for _ in range(num_chains):
            positions.append(
                [
                    value + rng.gauss(0.0, jitter_scale)
                    for value in data.true_log_volatility
                ]
            )
        return positions

    def log_prob(self, position: Sequence[float] | Any, data: Any | None = None) -> Any:
        """Evaluate the unnormalized latent stochastic-volatility log density."""
        if hasattr(position, "shape"):
            position_array = jnp.asarray(position)
            if position_array.shape != (self.dimension,):
                raise ValueError(
                    f"Expected position with shape ({self.dimension},), "
                    f"got {position_array.shape}"
                )
            observations = self._jax_data(data, dtype=position_array.dtype)
            initial_scale = self._stationary_scale()

            log_prob = _jax_normal_log_prob(position_array[0], 0.0, initial_scale)
            if self.dimension > 1:
                innovations = position_array[1:] - self.persistence * position_array[:-1]
                log_prob += jnp.sum(
                    _jax_normal_log_prob(innovations, 0.0, self.innovation_scale)
                )
            log_prob += jnp.sum(
                -0.5
                * (
                    observations * observations * jnp.exp(-position_array)
                    + position_array
                    + LOG_TWO_PI
                )
            )
            return log_prob

        if len(position) != self.dimension:
            raise ValueError(
                f"Expected position with dimension {self.dimension}, got {len(position)}"
            )
        observations = self._python_data(data).observations
        latent = [float(value) for value in position]
        initial_scale = self._stationary_scale()

        log_prob = _normal_log_prob(latent[0], 0.0, initial_scale)
        for previous, current in zip(latent[:-1], latent[1:], strict=False):
            log_prob += _normal_log_prob(
                current - self.persistence * previous,
                0.0,
                self.innovation_scale,
            )
        for observed, log_volatility in zip(observations, latent, strict=True):
            log_prob += _observation_log_prob(observed, log_volatility)
        return log_prob

    def _python_data(self, data: Any | None) -> StochasticVolatilityData:
        if data is None:
            return self._require_data()
        try:
            observations = tuple(float(value) for value in data["observations"])
            true_log_volatility = tuple(
                float(value)
                for value in data.get("true_log_volatility", (0.0,) * self.time_length)
            )
            record = StochasticVolatilityData(
                observations=observations,
                true_log_volatility=true_log_volatility,
                persistence=float(data.get("persistence", self.persistence)),
                innovation_scale=float(data.get("innovation_scale", self.innovation_scale)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "stochastic volatility data must contain numeric 'observations'"
            ) from exc
        self._validate_data(record)
        return record

    def _jax_data(self, data: Any | None, *, dtype: Any) -> Any:
        data_record = self._python_data(data)
        return jnp.asarray(data_record.observations, dtype=dtype)

    def _require_data(self) -> StochasticVolatilityData:
        if self.synthetic_data is None:
            raise ValueError("synthetic_data was not initialized")
        return self.synthetic_data

    def _stationary_scale(self) -> float:
        return self.innovation_scale / math.sqrt(1.0 - self.persistence * self.persistence)

    def _validate_data(self, data: StochasticVolatilityData | None) -> None:
        if data is None:
            raise ValueError("synthetic_data cannot be None")
        if len(data.observations) != self.time_length:
            raise ValueError("observations length must match time_length")
        if len(data.true_log_volatility) != self.time_length:
            raise ValueError("true_log_volatility length must match time_length")
        if not -1.0 < data.persistence < 1.0:
            raise ValueError("data persistence must be strictly between -1 and 1")
        if data.innovation_scale <= 0.0:
            raise ValueError("data innovation_scale must be positive")
        if any(not math.isfinite(value) for value in data.observations):
            raise ValueError("observations must be finite")
        if any(not math.isfinite(value) for value in data.true_log_volatility):
            raise ValueError("true_log_volatility values must be finite")


def stochastic_volatility_model(
    *,
    dimension: int,
    time_length: int | None = None,
    persistence: float = 0.95,
    innovation_scale: float = 0.2,
    data_seed: int = 29,
) -> StochasticVolatilityModel:
    """Construct the synthetic stochastic volatility benchmark model."""
    return StochasticVolatilityModel(
        dimension=dimension,
        time_length=time_length,
        persistence=persistence,
        innovation_scale=innovation_scale,
        data_seed=data_seed,
    )


def generate_synthetic_stochastic_volatility_data(
    *,
    time_length: int,
    persistence: float,
    innovation_scale: float,
    seed: int,
) -> StochasticVolatilityData:
    """Generate deterministic latent volatilities and observations."""
    if time_length <= 0:
        raise ValueError(f"time_length must be positive, got {time_length!r}")
    if not -1.0 < persistence < 1.0:
        raise ValueError(f"persistence must be strictly between -1 and 1, got {persistence!r}")
    if innovation_scale <= 0.0:
        raise ValueError(f"innovation_scale must be positive, got {innovation_scale!r}")

    rng = random.Random(int(seed))
    stationary_scale = innovation_scale / math.sqrt(1.0 - persistence * persistence)
    latent: list[float] = [rng.gauss(0.0, stationary_scale)]
    for _ in range(1, time_length):
        latent.append(persistence * latent[-1] + rng.gauss(0.0, innovation_scale))

    observations = tuple(rng.gauss(0.0, math.exp(0.5 * value)) for value in latent)
    return StochasticVolatilityData(
        observations=observations,
        true_log_volatility=tuple(latent),
        persistence=persistence,
        innovation_scale=innovation_scale,
    )


def _normal_log_prob(value: float, loc: float, scale: float) -> float:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * math.log(scale) + LOG_TWO_PI)


def _observation_log_prob(observed: float, log_volatility: float) -> float:
    return -0.5 * (
        observed * observed * math.exp(-log_volatility) + log_volatility + LOG_TWO_PI
    )


def _jax_normal_log_prob(value: Any, loc: Any, scale: Any) -> Any:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * jnp.log(scale) + LOG_TWO_PI)
