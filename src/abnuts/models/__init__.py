"""Benchmark model registry."""

from __future__ import annotations

from collections.abc import Callable

from abnuts.models.base import BenchmarkModel, ModelMetadata, vectorized_log_prob
from abnuts.models.eight_schools import (
    EightSchoolsModel,
    centered_eight_schools_model,
    noncentered_eight_schools_model,
)
from abnuts.models.funnel import FunnelModel
from abnuts.models.gaussian_process import (
    GaussianProcessModel,
    gaussian_process_model,
)
from abnuts.models.hierarchical_logistic import (
    HierarchicalLogisticModel,
    hierarchical_logistic_model,
)
from abnuts.models.stochastic_volatility import (
    StochasticVolatilityModel,
    stochastic_volatility_model,
)

ModelFactory = Callable[..., BenchmarkModel]

_MODEL_FACTORIES: dict[str, ModelFactory] = {
    "eight_schools_centered": centered_eight_schools_model,
    "eight_schools_noncentered": noncentered_eight_schools_model,
    "funnel": FunnelModel,
    "gaussian_process": gaussian_process_model,
    "hierarchical_logistic": hierarchical_logistic_model,
    "stochastic_volatility": stochastic_volatility_model,
}


def available_models() -> tuple[str, ...]:
    """Return the registered benchmark model names."""
    return tuple(sorted(_MODEL_FACTORIES))


def get_model(
    name: str,
    *,
    dimension: int,
    model_config: dict[str, object] | None = None,
) -> BenchmarkModel:
    """Construct a registered benchmark model by name."""
    try:
        factory = _MODEL_FACTORIES[name]
    except KeyError as exc:
        available = ", ".join(available_models())
        raise ValueError(f"Unknown model {name!r}. Available models: {available}") from exc
    return factory(dimension=dimension, **(model_config or {}))


__all__ = [
    "BenchmarkModel",
    "EightSchoolsModel",
    "FunnelModel",
    "GaussianProcessModel",
    "HierarchicalLogisticModel",
    "ModelMetadata",
    "StochasticVolatilityModel",
    "available_models",
    "get_model",
    "vectorized_log_prob",
]
