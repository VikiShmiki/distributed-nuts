"""Exact Gaussian-process regression benchmark target."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from abnuts.models.base import ModelMetadata

LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class GaussianProcessData:
    """Deterministic one-dimensional regression data for the GP benchmark."""

    inputs: tuple[float, ...]
    observations: tuple[float, ...]
    true_latent_function: tuple[float, ...]
    true_amplitude: float
    true_length_scale: float
    true_noise_scale: float

    @property
    def num_observations(self) -> int:
        """Return the number of regression observations."""
        return len(self.observations)


@dataclass(frozen=True)
class GaussianProcessModel:
    """Exact GP regression marginal likelihood over log hyperparameters."""

    dimension: int
    num_observations: int = 8
    data_seed: int = 37
    input_min: float = -1.0
    input_max: float = 1.0
    true_amplitude: float = 1.0
    true_length_scale: float = 0.35
    true_noise_scale: float = 0.08
    jitter: float = 1.0e-5
    log_amplitude_prior_mean: float = 0.0
    log_length_scale_prior_mean: float = math.log(0.5)
    log_noise_scale_prior_mean: float = math.log(0.1)
    log_hyperparameter_prior_std: float = 1.5
    synthetic_data: GaussianProcessData | None = None
    name: str = "gaussian_process"

    def __post_init__(self) -> None:
        """Validate dimensions and generate deterministic synthetic data."""
        if self.dimension != 3:
            raise ValueError(
                f"{self.name} requires dimension 3 "
                "(log_amplitude, log_length_scale, log_noise_scale); "
                f"got {self.dimension!r}"
            )
        if self.num_observations <= 0:
            raise ValueError(
                f"num_observations must be positive, got {self.num_observations!r}"
            )
        if self.input_min >= self.input_max:
            raise ValueError("input_min must be smaller than input_max")
        if self.true_amplitude <= 0.0:
            raise ValueError(f"true_amplitude must be positive, got {self.true_amplitude!r}")
        if self.true_length_scale <= 0.0:
            raise ValueError(
                f"true_length_scale must be positive, got {self.true_length_scale!r}"
            )
        if self.true_noise_scale <= 0.0:
            raise ValueError(
                f"true_noise_scale must be positive, got {self.true_noise_scale!r}"
            )
        if self.jitter < 0.0:
            raise ValueError(f"jitter must be non-negative, got {self.jitter!r}")
        if self.log_hyperparameter_prior_std <= 0.0:
            raise ValueError(
                "log_hyperparameter_prior_std must be positive, "
                f"got {self.log_hyperparameter_prior_std!r}"
            )

        data = self.synthetic_data
        if data is None:
            data = generate_synthetic_gaussian_process_data(
                num_observations=self.num_observations,
                seed=self.data_seed,
                input_min=self.input_min,
                input_max=self.input_max,
                true_amplitude=self.true_amplitude,
                true_length_scale=self.true_length_scale,
                true_noise_scale=self.true_noise_scale,
                jitter=self.jitter,
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
                "Exact one-dimensional Gaussian-process regression marginal "
                "likelihood with a squared-exponential kernel."
            ),
            extra={
                "model_family": "gaussian_process",
                "parameterization": "log_hyperparameters",
                "kernel": "squared_exponential",
                "input_dimension": 1,
                "num_observations": data.num_observations,
                "data_seed": self.data_seed,
                "jitter": self.jitter,
            },
        )

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate deterministic initial log-hyperparameter positions."""
        if num_chains <= 0:
            raise ValueError(f"num_chains must be positive, got {num_chains!r}")

        jitter_scale = float((config or {}).get("initial_jitter_scale", 0.03))
        if jitter_scale <= 0.0:
            raise ValueError(f"initial_jitter_scale must be positive, got {jitter_scale!r}")

        rng = random.Random(int(key))
        data = self._require_data()
        center = (
            math.log(data.true_amplitude),
            math.log(data.true_length_scale),
            math.log(data.true_noise_scale),
        )
        return [
            [value + rng.gauss(0.0, jitter_scale) for value in center]
            for _ in range(num_chains)
        ]

    def log_prob(self, position: Sequence[float] | Any, data: Any | None = None) -> Any:
        """Evaluate the exact GP regression log marginal likelihood plus priors."""
        if hasattr(position, "shape"):
            position_array = jnp.asarray(position)
            if position_array.shape != (self.dimension,):
                raise ValueError(
                    f"Expected position with shape ({self.dimension},), "
                    f"got {position_array.shape}"
                )
            inputs, observations = self._jax_data(data, dtype=position_array.dtype)
            amplitude = jnp.exp(position_array[0])
            length_scale = jnp.exp(position_array[1])
            noise_scale = jnp.exp(position_array[2])
            log_prob = _jax_gp_log_marginal_likelihood(
                inputs=inputs,
                observations=observations,
                amplitude=amplitude,
                length_scale=length_scale,
                noise_scale=noise_scale,
                jitter=jnp.asarray(self.jitter, dtype=position_array.dtype),
            )
            log_prob += _jax_normal_log_prob(
                position_array[0],
                self.log_amplitude_prior_mean,
                self.log_hyperparameter_prior_std,
            )
            log_prob += _jax_normal_log_prob(
                position_array[1],
                self.log_length_scale_prior_mean,
                self.log_hyperparameter_prior_std,
            )
            log_prob += _jax_normal_log_prob(
                position_array[2],
                self.log_noise_scale_prior_mean,
                self.log_hyperparameter_prior_std,
            )
            return log_prob

        if len(position) != self.dimension:
            raise ValueError(
                f"Expected position with dimension {self.dimension}, got {len(position)}"
            )

        log_amplitude, log_length_scale, log_noise_scale = (
            _finite_float(value, name)
            for value, name in zip(
                position,
                ("log_amplitude", "log_length_scale", "log_noise_scale"),
                strict=True,
            )
        )
        data_record = self._python_data(data)
        amplitude = _positive_exp(log_amplitude, "log_amplitude")
        length_scale = _positive_exp(log_length_scale, "log_length_scale")
        noise_scale = _positive_exp(log_noise_scale, "log_noise_scale")
        log_prob = _gp_log_marginal_likelihood(
            inputs=data_record.inputs,
            observations=data_record.observations,
            amplitude=amplitude,
            length_scale=length_scale,
            noise_scale=noise_scale,
            jitter=self.jitter,
        )
        log_prob += _normal_log_prob(
            log_amplitude,
            self.log_amplitude_prior_mean,
            self.log_hyperparameter_prior_std,
        )
        log_prob += _normal_log_prob(
            log_length_scale,
            self.log_length_scale_prior_mean,
            self.log_hyperparameter_prior_std,
        )
        log_prob += _normal_log_prob(
            log_noise_scale,
            self.log_noise_scale_prior_mean,
            self.log_hyperparameter_prior_std,
        )
        return log_prob

    def _python_data(self, data: Any | None) -> GaussianProcessData:
        if data is None:
            return self._require_data()
        try:
            inputs = tuple(float(value) for value in data["inputs"])
            observations = tuple(float(value) for value in data["observations"])
            record = GaussianProcessData(
                inputs=inputs,
                observations=observations,
                true_latent_function=tuple(
                    float(value)
                    for value in data.get("true_latent_function", (0.0,) * len(inputs))
                ),
                true_amplitude=float(data.get("true_amplitude", self.true_amplitude)),
                true_length_scale=float(
                    data.get("true_length_scale", self.true_length_scale)
                ),
                true_noise_scale=float(data.get("true_noise_scale", self.true_noise_scale)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Gaussian-process data must contain numeric 'inputs' and "
                "'observations' fields"
            ) from exc
        self._validate_data(record)
        return record

    def _jax_data(self, data: Any | None, *, dtype: Any) -> tuple[Any, Any]:
        data_record = self._python_data(data)
        return (
            jnp.asarray(data_record.inputs, dtype=dtype),
            jnp.asarray(data_record.observations, dtype=dtype),
        )

    def _require_data(self) -> GaussianProcessData:
        if self.synthetic_data is None:
            raise ValueError("synthetic_data was not initialized")
        return self.synthetic_data

    def _validate_data(self, data: GaussianProcessData | None) -> None:
        if data is None:
            raise ValueError("synthetic_data cannot be None")
        if len(data.inputs) != self.num_observations:
            raise ValueError("inputs length must match num_observations")
        if len(data.observations) != self.num_observations:
            raise ValueError("observations length must match num_observations")
        if len(data.true_latent_function) != self.num_observations:
            raise ValueError("true_latent_function length must match num_observations")
        if data.true_amplitude <= 0.0:
            raise ValueError("data true_amplitude must be positive")
        if data.true_length_scale <= 0.0:
            raise ValueError("data true_length_scale must be positive")
        if data.true_noise_scale <= 0.0:
            raise ValueError("data true_noise_scale must be positive")
        if any(not math.isfinite(value) for value in data.inputs):
            raise ValueError("inputs must be finite")
        if any(not math.isfinite(value) for value in data.observations):
            raise ValueError("observations must be finite")
        if any(not math.isfinite(value) for value in data.true_latent_function):
            raise ValueError("true_latent_function values must be finite")


def gaussian_process_model(
    *,
    dimension: int,
    num_observations: int = 8,
    data_seed: int = 37,
    input_min: float = -1.0,
    input_max: float = 1.0,
    true_amplitude: float = 1.0,
    true_length_scale: float = 0.35,
    true_noise_scale: float = 0.08,
    jitter: float = 1.0e-5,
) -> GaussianProcessModel:
    """Construct the exact GP regression benchmark model."""
    return GaussianProcessModel(
        dimension=dimension,
        num_observations=num_observations,
        data_seed=data_seed,
        input_min=input_min,
        input_max=input_max,
        true_amplitude=true_amplitude,
        true_length_scale=true_length_scale,
        true_noise_scale=true_noise_scale,
        jitter=jitter,
    )


def generate_synthetic_gaussian_process_data(
    *,
    num_observations: int,
    seed: int,
    input_min: float = -1.0,
    input_max: float = 1.0,
    true_amplitude: float = 1.0,
    true_length_scale: float = 0.35,
    true_noise_scale: float = 0.08,
    jitter: float = 1.0e-5,
) -> GaussianProcessData:
    """Generate deterministic one-dimensional GP regression observations."""
    if num_observations <= 0:
        raise ValueError(f"num_observations must be positive, got {num_observations!r}")
    if input_min >= input_max:
        raise ValueError("input_min must be smaller than input_max")
    if true_amplitude <= 0.0:
        raise ValueError(f"true_amplitude must be positive, got {true_amplitude!r}")
    if true_length_scale <= 0.0:
        raise ValueError(
            f"true_length_scale must be positive, got {true_length_scale!r}"
        )
    if true_noise_scale <= 0.0:
        raise ValueError(f"true_noise_scale must be positive, got {true_noise_scale!r}")
    if jitter < 0.0:
        raise ValueError(f"jitter must be non-negative, got {jitter!r}")

    rng = np.random.default_rng(int(seed))
    inputs = np.linspace(input_min, input_max, num_observations, dtype=np.float64)
    covariance = _numpy_squared_exponential_covariance(
        inputs,
        amplitude=true_amplitude,
        length_scale=true_length_scale,
    )
    covariance = covariance + jitter * np.eye(num_observations, dtype=np.float64)
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "GP synthetic data Cholesky failed; increase jitter or check kernel "
            "hyperparameters."
        ) from exc

    latent = chol @ rng.standard_normal(num_observations)
    observations = latent + rng.normal(0.0, true_noise_scale, size=num_observations)
    return GaussianProcessData(
        inputs=tuple(float(value) for value in inputs),
        observations=tuple(float(value) for value in observations),
        true_latent_function=tuple(float(value) for value in latent),
        true_amplitude=true_amplitude,
        true_length_scale=true_length_scale,
        true_noise_scale=true_noise_scale,
    )


def _gp_log_marginal_likelihood(
    *,
    inputs: tuple[float, ...],
    observations: tuple[float, ...],
    amplitude: float,
    length_scale: float,
    noise_scale: float,
    jitter: float,
) -> float:
    input_array = np.asarray(inputs, dtype=np.float64)
    observation_array = np.asarray(observations, dtype=np.float64)
    covariance = _numpy_squared_exponential_covariance(
        input_array,
        amplitude=amplitude,
        length_scale=length_scale,
    )
    covariance = covariance + (noise_scale * noise_scale + jitter) * np.eye(
        input_array.shape[0],
        dtype=np.float64,
    )
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "GP covariance Cholesky failed; increase jitter or check "
            "log hyperparameters."
        ) from exc
    alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, observation_array))
    log_det = 2.0 * float(np.sum(np.log(np.diag(chol))))
    quadratic = float(observation_array @ alpha)
    num_observations = observation_array.shape[0]
    return -0.5 * (quadratic + log_det + num_observations * LOG_TWO_PI)


def _jax_gp_log_marginal_likelihood(
    *,
    inputs: Any,
    observations: Any,
    amplitude: Any,
    length_scale: Any,
    noise_scale: Any,
    jitter: Any,
) -> Any:
    distance = inputs[:, None] - inputs[None, :]
    covariance = amplitude * amplitude * jnp.exp(
        -0.5 * (distance / length_scale) ** 2
    )
    covariance += (noise_scale * noise_scale + jitter) * jnp.eye(
        inputs.shape[0],
        dtype=inputs.dtype,
    )
    chol = jnp.linalg.cholesky(covariance)
    alpha = jnp.linalg.solve(chol.T, jnp.linalg.solve(chol, observations))
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
    quadratic = observations @ alpha
    return -0.5 * (quadratic + log_det + observations.shape[0] * LOG_TWO_PI)


def _numpy_squared_exponential_covariance(
    inputs: np.ndarray,
    *,
    amplitude: float,
    length_scale: float,
) -> np.ndarray:
    distance = inputs[:, None] - inputs[None, :]
    return amplitude * amplitude * np.exp(-0.5 * (distance / length_scale) ** 2)


def _finite_float(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {parsed!r}")
    return parsed


def _positive_exp(value: float, name: str) -> float:
    try:
        parsed = math.exp(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exponentiated to a non-finite value") from exc
    if parsed <= 0.0 or not math.isfinite(parsed):
        raise ValueError(f"{name} exponentiated to a non-positive or non-finite value")
    return parsed


def _normal_log_prob(value: float, loc: float, scale: float) -> float:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * math.log(scale) + LOG_TWO_PI)


def _jax_normal_log_prob(value: Any, loc: Any, scale: Any) -> Any:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * jnp.log(scale) + LOG_TWO_PI)
