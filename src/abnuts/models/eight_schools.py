"""Centered and non-centered Eight Schools benchmark targets."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from abnuts.models.base import ModelMetadata

LOG_TWO_PI = math.log(2.0 * math.pi)
DEFAULT_TREATMENT_EFFECTS = (28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0)
DEFAULT_STANDARD_ERRORS = (15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0)
PARAMETERIZATIONS = frozenset({"centered", "noncentered"})


@dataclass(frozen=True)
class EightSchoolsModel:
    """Eight Schools hierarchical normal model in one unconstrained parameterization."""

    dimension: int
    parameterization: str
    name: str
    treatment_effects: tuple[float, ...] = DEFAULT_TREATMENT_EFFECTS
    standard_errors: tuple[float, ...] = DEFAULT_STANDARD_ERRORS
    mu_prior_std: float = 5.0
    tau_prior_scale: float = 5.0

    def __post_init__(self) -> None:
        """Validate parameterization, data, and unconstrained dimension."""
        if self.parameterization not in PARAMETERIZATIONS:
            valid = ", ".join(sorted(PARAMETERIZATIONS))
            raise ValueError(
                f"parameterization must be one of {valid}; got {self.parameterization!r}"
            )
        if len(self.treatment_effects) != len(self.standard_errors):
            raise ValueError("treatment_effects and standard_errors must have the same length")
        if not self.treatment_effects:
            raise ValueError("Eight Schools requires at least one school")
        if any(error <= 0.0 for error in self.standard_errors):
            raise ValueError("all Eight Schools standard errors must be positive")
        if self.mu_prior_std <= 0.0:
            raise ValueError(f"mu_prior_std must be positive, got {self.mu_prior_std!r}")
        if self.tau_prior_scale <= 0.0:
            raise ValueError(f"tau_prior_scale must be positive, got {self.tau_prior_scale!r}")

        expected_dimension = len(self.treatment_effects) + 2
        if self.dimension != expected_dimension:
            raise ValueError(
                f"{self.name} requires dimension {expected_dimension} "
                f"(mu, log_tau, and {len(self.treatment_effects)} school parameters); "
                f"got {self.dimension!r}"
            )

    @property
    def metadata(self) -> ModelMetadata:
        """Return serializable metadata for result manifests."""
        return ModelMetadata(
            name=self.name,
            dimension=self.dimension,
            event_shape=(self.dimension,),
            description=(
                "Eight Schools hierarchical normal benchmark with an unconstrained "
                f"{self.parameterization} parameterization."
            ),
            extra={
                "model_family": "eight_schools",
                "parameterization": self.parameterization,
                "num_schools": len(self.treatment_effects),
            },
        )

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate deterministic initial positions near the observed data scale."""
        if num_chains <= 0:
            raise ValueError(f"num_chains must be positive, got {num_chains!r}")

        jitter_scale = float((config or {}).get("initial_jitter_scale", 0.05))
        if jitter_scale <= 0.0:
            raise ValueError(f"initial_jitter_scale must be positive, got {jitter_scale!r}")

        rng = random.Random(int(key))
        mu_start = sum(self.treatment_effects) / len(self.treatment_effects)
        tau_start = self.tau_prior_scale
        log_tau_start = math.log(tau_start)

        positions: list[list[float]] = []
        for _ in range(num_chains):
            mu = mu_start + rng.gauss(0.0, jitter_scale)
            log_tau = log_tau_start + rng.gauss(0.0, jitter_scale)
            if self.parameterization == "centered":
                school_parameters = [
                    effect + rng.gauss(0.0, jitter_scale)
                    for effect in self.treatment_effects
                ]
            else:
                school_parameters = [
                    ((effect - mu_start) / tau_start) + rng.gauss(0.0, jitter_scale)
                    for effect in self.treatment_effects
                ]
            positions.append([mu, log_tau, *school_parameters])
        return positions

    def log_prob(self, position: Sequence[float] | Any, data: Any | None = None) -> Any:
        """Evaluate the unnormalized Eight Schools log density."""
        if hasattr(position, "shape"):
            position_array = jnp.asarray(position)
            if position_array.shape != (self.dimension,):
                raise ValueError(
                    f"Expected position with shape ({self.dimension},), "
                    f"got {position_array.shape}"
                )
            y, sigma = self._jax_data(data, dtype=position_array.dtype)
            mu = position_array[0]
            log_tau = position_array[1]
            tau = jnp.exp(log_tau)
            latent = position_array[2:]

            log_prob = _jax_normal_log_prob(mu, 0.0, self.mu_prior_std)
            log_prob += _jax_half_cauchy_log_prob(tau, self.tau_prior_scale) + log_tau
            if self.parameterization == "centered":
                theta = latent
                log_prob += jnp.sum(_jax_normal_log_prob(theta, mu, tau))
            else:
                theta = mu + tau * latent
                log_prob += jnp.sum(_jax_standard_normal_log_prob(latent))
            log_prob += jnp.sum(_jax_normal_log_prob(y, theta, sigma))
            return log_prob

        if len(position) != self.dimension:
            raise ValueError(
                f"Expected position with dimension {self.dimension}, got {len(position)}"
            )
        y, sigma = self._python_data(data)
        mu = float(position[0])
        log_tau = float(position[1])
        tau = math.exp(log_tau)
        latent = [float(value) for value in position[2:]]

        log_prob = _normal_log_prob(mu, 0.0, self.mu_prior_std)
        log_prob += _half_cauchy_log_prob(tau, self.tau_prior_scale) + log_tau
        if self.parameterization == "centered":
            theta = latent
            log_prob += sum(_normal_log_prob(value, mu, tau) for value in latent)
        else:
            theta = [mu + tau * value for value in latent]
            log_prob += sum(_standard_normal_log_prob(value) for value in latent)
        log_prob += sum(
            _normal_log_prob(observed, school_theta, error)
            for observed, school_theta, error in zip(y, theta, sigma, strict=True)
        )
        return log_prob

    def _python_data(self, data: Any | None) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if data is None:
            return self.treatment_effects, self.standard_errors
        try:
            y = tuple(float(value) for value in data["y"])
            sigma = tuple(float(value) for value in data["sigma"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Eight Schools data must contain numeric 'y' and 'sigma' fields"
            ) from exc
        if len(y) != len(self.treatment_effects) or len(sigma) != len(self.standard_errors):
            raise ValueError(
                f"Eight Schools data must contain {len(self.treatment_effects)} schools"
            )
        if any(error <= 0.0 for error in sigma):
            raise ValueError("Eight Schools data standard errors must be positive")
        return y, sigma

    def _jax_data(self, data: Any | None, *, dtype: Any) -> tuple[Any, Any]:
        y, sigma = self._python_data(data)
        return jnp.asarray(y, dtype=dtype), jnp.asarray(sigma, dtype=dtype)


def centered_eight_schools_model(*, dimension: int) -> EightSchoolsModel:
    """Construct the centered Eight Schools benchmark model."""
    return EightSchoolsModel(
        dimension=dimension,
        parameterization="centered",
        name="eight_schools_centered",
    )


def noncentered_eight_schools_model(*, dimension: int) -> EightSchoolsModel:
    """Construct the non-centered Eight Schools benchmark model."""
    return EightSchoolsModel(
        dimension=dimension,
        parameterization="noncentered",
        name="eight_schools_noncentered",
    )


def _normal_log_prob(value: float, loc: float, scale: float) -> float:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * math.log(scale) + LOG_TWO_PI)


def _standard_normal_log_prob(value: float) -> float:
    return -0.5 * (value * value + LOG_TWO_PI)


def _half_cauchy_log_prob(value: float, scale: float) -> float:
    return math.log(2.0 / (math.pi * scale)) - math.log1p((value / scale) ** 2)


def _jax_normal_log_prob(value: Any, loc: Any, scale: Any) -> Any:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * jnp.log(scale) + LOG_TWO_PI)


def _jax_standard_normal_log_prob(value: Any) -> Any:
    return -0.5 * (value * value + LOG_TWO_PI)


def _jax_half_cauchy_log_prob(value: Any, scale: float) -> Any:
    return math.log(2.0 / (math.pi * scale)) - jnp.log1p((value / scale) ** 2)
