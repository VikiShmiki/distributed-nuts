"""Synthetic hierarchical logistic regression benchmark target."""

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
class SyntheticHierarchicalLogisticData:
    """Deterministic grouped binary data for the hierarchical logistic target."""

    group_index: tuple[int, ...]
    features: tuple[tuple[float, ...], ...]
    outcomes: tuple[int, ...]
    true_intercept: float
    true_group_scale: float
    true_group_offsets: tuple[float, ...]
    true_coefficients: tuple[float, ...]

    @property
    def num_observations(self) -> int:
        """Return the number of binary observations."""
        return len(self.outcomes)


@dataclass(frozen=True)
class HierarchicalLogisticModel:
    """Non-centered hierarchical logistic regression with synthetic grouped data."""

    dimension: int
    num_groups: int = 4
    observations_per_group: int = 3
    num_features: int | None = None
    data_seed: int = 17
    feature_scale: float = 1.0
    intercept_prior_std: float = 2.5
    coefficient_prior_std: float = 1.5
    group_scale_prior_scale: float = 1.0
    synthetic_data: SyntheticHierarchicalLogisticData | None = None
    name: str = "hierarchical_logistic"

    def __post_init__(self) -> None:
        """Validate dimensions and generate deterministic synthetic data."""
        if self.num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {self.num_groups!r}")
        if self.observations_per_group <= 0:
            raise ValueError(
                "observations_per_group must be positive, "
                f"got {self.observations_per_group!r}"
            )
        if self.feature_scale <= 0.0:
            raise ValueError(f"feature_scale must be positive, got {self.feature_scale!r}")
        if self.intercept_prior_std <= 0.0:
            raise ValueError(
                f"intercept_prior_std must be positive, got {self.intercept_prior_std!r}"
            )
        if self.coefficient_prior_std <= 0.0:
            raise ValueError(
                "coefficient_prior_std must be positive, "
                f"got {self.coefficient_prior_std!r}"
            )
        if self.group_scale_prior_scale <= 0.0:
            raise ValueError(
                "group_scale_prior_scale must be positive, "
                f"got {self.group_scale_prior_scale!r}"
            )

        if self.num_features is None:
            num_features = self.dimension - self.num_groups - 2
        else:
            num_features = int(self.num_features)
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features!r}")

        expected_dimension = 2 + self.num_groups + num_features
        if self.dimension != expected_dimension:
            raise ValueError(
                f"{self.name} requires dimension {expected_dimension} "
                f"(intercept, log_group_scale, {self.num_groups} group offsets, "
                f"and {num_features} coefficients); got {self.dimension!r}"
            )
        object.__setattr__(self, "num_features", num_features)

        data = self.synthetic_data
        if data is None:
            data = generate_synthetic_hierarchical_logistic_data(
                num_groups=self.num_groups,
                observations_per_group=self.observations_per_group,
                num_features=num_features,
                seed=self.data_seed,
                feature_scale=self.feature_scale,
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
                "Synthetic hierarchical logistic regression with non-centered "
                "group effects and deterministic generated binary observations."
            ),
            extra={
                "model_family": "hierarchical_logistic",
                "parameterization": "noncentered_group_effects",
                "num_groups": self.num_groups,
                "observations_per_group": self.observations_per_group,
                "num_observations": data.num_observations,
                "num_features": self.num_features,
                "data_seed": self.data_seed,
            },
        )

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate deterministic initial positions near the synthetic truth."""
        if num_chains <= 0:
            raise ValueError(f"num_chains must be positive, got {num_chains!r}")

        jitter_scale = float((config or {}).get("initial_jitter_scale", 0.05))
        if jitter_scale <= 0.0:
            raise ValueError(f"initial_jitter_scale must be positive, got {jitter_scale!r}")

        rng = random.Random(int(key))
        data = self._require_data()
        log_group_scale = math.log(data.true_group_scale)
        positions: list[list[float]] = []
        for _ in range(num_chains):
            intercept = data.true_intercept + rng.gauss(0.0, jitter_scale)
            scale = log_group_scale + rng.gauss(0.0, jitter_scale)
            group_offsets = [
                value + rng.gauss(0.0, jitter_scale) for value in data.true_group_offsets
            ]
            coefficients = [
                value + rng.gauss(0.0, jitter_scale) for value in data.true_coefficients
            ]
            positions.append([intercept, scale, *group_offsets, *coefficients])
        return positions

    def log_prob(self, position: Sequence[float] | Any, data: Any | None = None) -> Any:
        """Evaluate the unnormalized synthetic hierarchical logistic log density."""
        if hasattr(position, "shape"):
            position_array = jnp.asarray(position)
            if position_array.shape != (self.dimension,):
                raise ValueError(
                    f"Expected position with shape ({self.dimension},), "
                    f"got {position_array.shape}"
                )
            group_index, features, outcomes = self._jax_data(data, dtype=position_array.dtype)
            intercept = position_array[0]
            log_group_scale = position_array[1]
            group_offsets = position_array[2 : 2 + self.num_groups]
            coefficients = position_array[2 + self.num_groups :]
            group_scale = jnp.exp(log_group_scale)
            group_effects = group_scale * group_offsets
            logits = intercept + group_effects[group_index] + features @ coefficients

            log_prob = _jax_normal_log_prob(intercept, 0.0, self.intercept_prior_std)
            log_prob += _jax_half_normal_log_prob(
                group_scale,
                self.group_scale_prior_scale,
            )
            log_prob += log_group_scale
            log_prob += jnp.sum(_jax_standard_normal_log_prob(group_offsets))
            log_prob += jnp.sum(
                _jax_normal_log_prob(coefficients, 0.0, self.coefficient_prior_std)
            )
            log_prob += jnp.sum(_jax_bernoulli_logit_log_prob(outcomes, logits))
            return log_prob

        if len(position) != self.dimension:
            raise ValueError(
                f"Expected position with dimension {self.dimension}, got {len(position)}"
            )
        data_record = self._python_data(data)
        intercept = float(position[0])
        log_group_scale = float(position[1])
        group_offsets = [float(value) for value in position[2 : 2 + self.num_groups]]
        coefficients = [float(value) for value in position[2 + self.num_groups :]]
        group_scale = math.exp(log_group_scale)
        group_effects = [group_scale * value for value in group_offsets]

        log_prob = _normal_log_prob(intercept, 0.0, self.intercept_prior_std)
        log_prob += _half_normal_log_prob(group_scale, self.group_scale_prior_scale)
        log_prob += log_group_scale
        log_prob += sum(_standard_normal_log_prob(value) for value in group_offsets)
        log_prob += sum(
            _normal_log_prob(value, 0.0, self.coefficient_prior_std)
            for value in coefficients
        )
        for group, features, outcome in zip(
            data_record.group_index,
            data_record.features,
            data_record.outcomes,
            strict=True,
        ):
            logit = intercept + group_effects[group] + sum(
                feature * coefficient
                for feature, coefficient in zip(features, coefficients, strict=True)
            )
            log_prob += _bernoulli_logit_log_prob(outcome, logit)
        return log_prob

    def _python_data(self, data: Any | None) -> SyntheticHierarchicalLogisticData:
        if data is None:
            return self._require_data()
        try:
            record = SyntheticHierarchicalLogisticData(
                group_index=tuple(int(value) for value in data["group_index"]),
                features=tuple(
                    tuple(float(feature) for feature in row) for row in data["features"]
                ),
                outcomes=tuple(int(value) for value in data["outcomes"]),
                true_intercept=float(data.get("true_intercept", 0.0)),
                true_group_scale=float(data.get("true_group_scale", 1.0)),
                true_group_offsets=tuple(
                    float(value)
                    for value in data.get("true_group_offsets", (0.0,) * self.num_groups)
                ),
                true_coefficients=tuple(
                    float(value)
                    for value in data.get("true_coefficients", (0.0,) * self.num_features)
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "hierarchical logistic data must contain numeric 'group_index', "
                "'features', and binary 'outcomes' fields"
            ) from exc
        self._validate_data(record)
        return record

    def _jax_data(self, data: Any | None, *, dtype: Any) -> tuple[Any, Any, Any]:
        data_record = self._python_data(data)
        return (
            jnp.asarray(data_record.group_index, dtype=jnp.int32),
            jnp.asarray(data_record.features, dtype=dtype),
            jnp.asarray(data_record.outcomes, dtype=dtype),
        )

    def _require_data(self) -> SyntheticHierarchicalLogisticData:
        if self.synthetic_data is None:
            raise ValueError("synthetic_data was not initialized")
        return self.synthetic_data

    def _validate_data(self, data: SyntheticHierarchicalLogisticData | None) -> None:
        if data is None:
            raise ValueError("synthetic_data cannot be None")
        if len(data.group_index) != len(data.features) or len(data.outcomes) != len(
            data.group_index
        ):
            raise ValueError("group_index, features, and outcomes must have equal length")
        if len(data.group_index) != self.num_groups * self.observations_per_group:
            raise ValueError(
                "synthetic data size must equal num_groups * observations_per_group"
            )
        if len(data.true_group_offsets) != self.num_groups:
            raise ValueError("true_group_offsets length must match num_groups")
        if len(data.true_coefficients) != self.num_features:
            raise ValueError("true_coefficients length must match num_features")
        if data.true_group_scale <= 0.0:
            raise ValueError("true_group_scale must be positive")
        for group in data.group_index:
            if group < 0 or group >= self.num_groups:
                raise ValueError(f"group index {group!r} is outside [0, {self.num_groups})")
        for row in data.features:
            if len(row) != self.num_features:
                raise ValueError("all feature rows must match num_features")
        if any(outcome not in (0, 1) for outcome in data.outcomes):
            raise ValueError("outcomes must be binary values 0 or 1")


def hierarchical_logistic_model(
    *,
    dimension: int,
    num_groups: int = 4,
    observations_per_group: int = 3,
    num_features: int | None = None,
    data_seed: int = 17,
    feature_scale: float = 1.0,
) -> HierarchicalLogisticModel:
    """Construct the synthetic hierarchical logistic benchmark model."""
    return HierarchicalLogisticModel(
        dimension=dimension,
        num_groups=num_groups,
        observations_per_group=observations_per_group,
        num_features=num_features,
        data_seed=data_seed,
        feature_scale=feature_scale,
    )


def generate_synthetic_hierarchical_logistic_data(
    *,
    num_groups: int,
    observations_per_group: int,
    num_features: int,
    seed: int,
    feature_scale: float = 1.0,
) -> SyntheticHierarchicalLogisticData:
    """Generate deterministic grouped binary observations for smoke benchmarks."""
    if num_groups <= 0:
        raise ValueError(f"num_groups must be positive, got {num_groups!r}")
    if observations_per_group <= 0:
        raise ValueError(
            f"observations_per_group must be positive, got {observations_per_group!r}"
        )
    if num_features <= 0:
        raise ValueError(f"num_features must be positive, got {num_features!r}")
    if feature_scale <= 0.0:
        raise ValueError(f"feature_scale must be positive, got {feature_scale!r}")

    rng = random.Random(int(seed))
    true_intercept = rng.gauss(-0.25, 0.25)
    true_group_scale = 0.7 + 0.2 * rng.random()
    true_group_offsets = tuple(rng.gauss(0.0, 1.0) for _ in range(num_groups))
    true_coefficients = tuple(rng.gauss(0.0, 0.6) for _ in range(num_features))

    group_index: list[int] = []
    features: list[tuple[float, ...]] = []
    outcomes: list[int] = []
    for group in range(num_groups):
        group_effect = true_group_scale * true_group_offsets[group]
        for _ in range(observations_per_group):
            row = tuple(rng.gauss(0.0, feature_scale) for _ in range(num_features))
            logit = true_intercept + group_effect + sum(
                value * coefficient
                for value, coefficient in zip(row, true_coefficients, strict=True)
            )
            probability = _sigmoid(logit)
            outcome = 1 if rng.random() < probability else 0
            group_index.append(group)
            features.append(row)
            outcomes.append(outcome)

    return SyntheticHierarchicalLogisticData(
        group_index=tuple(group_index),
        features=tuple(features),
        outcomes=tuple(outcomes),
        true_intercept=true_intercept,
        true_group_scale=true_group_scale,
        true_group_offsets=true_group_offsets,
        true_coefficients=true_coefficients,
    )


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        negative = math.exp(-value)
        return 1.0 / (1.0 + negative)
    positive = math.exp(value)
    return positive / (1.0 + positive)


def _normal_log_prob(value: float, loc: float, scale: float) -> float:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * math.log(scale) + LOG_TWO_PI)


def _standard_normal_log_prob(value: float) -> float:
    return -0.5 * (value * value + LOG_TWO_PI)


def _half_normal_log_prob(value: float, scale: float) -> float:
    return math.log(2.0) + _normal_log_prob(value, 0.0, scale)


def _log_sigmoid(value: float) -> float:
    if value >= 0.0:
        return -math.log1p(math.exp(-value))
    return value - math.log1p(math.exp(value))


def _bernoulli_logit_log_prob(outcome: int, logit: float) -> float:
    if outcome == 1:
        return _log_sigmoid(logit)
    return _log_sigmoid(-logit)


def _jax_normal_log_prob(value: Any, loc: Any, scale: Any) -> Any:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * jnp.log(scale) + LOG_TWO_PI)


def _jax_standard_normal_log_prob(value: Any) -> Any:
    return -0.5 * (value * value + LOG_TWO_PI)


def _jax_half_normal_log_prob(value: Any, scale: float) -> Any:
    return math.log(2.0) + _jax_normal_log_prob(value, 0.0, scale)


def _jax_bernoulli_logit_log_prob(outcome: Any, logit: Any) -> Any:
    return outcome * -jnp.logaddexp(0.0, -logit) + (1.0 - outcome) * -jnp.logaddexp(
        0.0,
        logit,
    )
