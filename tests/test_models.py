from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from abnuts.models import available_models, get_model, vectorized_log_prob
from abnuts.models.eight_schools import (
    EightSchoolsModel,
    centered_eight_schools_model,
    noncentered_eight_schools_model,
)
from abnuts.models.funnel import FunnelModel
from abnuts.models.gaussian_process import (
    GaussianProcessData,
    GaussianProcessModel,
    gaussian_process_model,
    generate_synthetic_gaussian_process_data,
)
from abnuts.models.hierarchical_logistic import (
    HierarchicalLogisticModel,
    generate_synthetic_hierarchical_logistic_data,
    hierarchical_logistic_model,
)
from abnuts.models.stochastic_volatility import (
    StochasticVolatilityModel,
    generate_synthetic_stochastic_volatility_data,
    stochastic_volatility_model,
)


def test_registry_loads_funnel_with_configurable_dimension() -> None:
    model = get_model("funnel", dimension=6)

    assert "funnel" in available_models()
    assert model.name == "funnel"
    assert model.dimension == 6
    assert model.metadata.as_dict()["event_shape"] == [6]


def test_funnel_initial_positions_are_vectorized_and_deterministic() -> None:
    model = FunnelModel(dimension=4)

    positions = model.initial_position(key=17, num_chains=3)
    repeated = model.initial_position(key=17, num_chains=3)

    assert positions == repeated
    assert len(positions) == 3
    assert all(len(position) == 4 for position in positions)


def test_funnel_log_prob_is_finite_for_generated_initial_positions() -> None:
    model = FunnelModel(dimension=5)
    positions = model.initial_position(key=0, num_chains=8)

    log_probs = vectorized_log_prob(model, positions)

    assert len(log_probs) == 8
    assert all(math.isfinite(value) for value in log_probs)


def test_funnel_rejects_invalid_dimension() -> None:
    with pytest.raises(ValueError, match="dimension >= 2"):
        FunnelModel(dimension=1)


@pytest.mark.parametrize(
    ("model_name", "parameterization"),
    (
        ("eight_schools_centered", "centered"),
        ("eight_schools_noncentered", "noncentered"),
    ),
)
def test_registry_loads_eight_schools_variants(
    model_name: str,
    parameterization: str,
) -> None:
    model = get_model(model_name, dimension=10)
    metadata = model.metadata.as_dict()

    assert model_name in available_models()
    assert model.name == model_name
    assert model.dimension == 10
    assert metadata["event_shape"] == [10]
    assert metadata["model_family"] == "eight_schools"
    assert metadata["parameterization"] == parameterization
    assert metadata["num_schools"] == 8


@pytest.mark.parametrize(
    "model",
    (
        centered_eight_schools_model(dimension=10),
        noncentered_eight_schools_model(dimension=10),
    ),
)
def test_eight_schools_initial_positions_are_deterministic(
    model: EightSchoolsModel,
) -> None:
    positions = model.initial_position(key=23, num_chains=3)
    repeated = model.initial_position(key=23, num_chains=3)

    assert positions == repeated
    assert len(positions) == 3
    assert all(len(position) == 10 for position in positions)


@pytest.mark.parametrize(
    "model",
    (
        centered_eight_schools_model(dimension=10),
        noncentered_eight_schools_model(dimension=10),
    ),
)
def test_eight_schools_log_prob_and_gradient_are_finite(
    model: EightSchoolsModel,
) -> None:
    positions = model.initial_position(key=0, num_chains=4)
    log_probs = vectorized_log_prob(model, positions)

    assert len(log_probs) == 4
    assert all(math.isfinite(value) for value in log_probs)

    position = jnp.asarray(positions[0], dtype=jnp.float32)
    value, grad = jax.value_and_grad(lambda q: model.log_prob(q))(position)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_eight_schools_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="requires dimension 10"):
        centered_eight_schools_model(dimension=9)


def test_registry_loads_hierarchical_logistic_with_group_metadata() -> None:
    model = get_model(
        "hierarchical_logistic",
        dimension=8,
        model_config={
            "num_groups": 4,
            "observations_per_group": 3,
            "num_features": 2,
            "data_seed": 17,
        },
    )
    metadata = model.metadata.as_dict()

    assert "hierarchical_logistic" in available_models()
    assert model.name == "hierarchical_logistic"
    assert model.dimension == 8
    assert metadata["event_shape"] == [8]
    assert metadata["model_family"] == "hierarchical_logistic"
    assert metadata["num_groups"] == 4
    assert metadata["observations_per_group"] == 3
    assert metadata["num_observations"] == 12
    assert metadata["num_features"] == 2


def test_hierarchical_logistic_synthetic_data_is_deterministic() -> None:
    data = generate_synthetic_hierarchical_logistic_data(
        num_groups=3,
        observations_per_group=4,
        num_features=2,
        seed=19,
    )
    repeated = generate_synthetic_hierarchical_logistic_data(
        num_groups=3,
        observations_per_group=4,
        num_features=2,
        seed=19,
    )

    assert data == repeated
    assert data.num_observations == 12
    assert len(data.group_index) == 12
    assert all(outcome in (0, 1) for outcome in data.outcomes)


def test_hierarchical_logistic_initial_positions_are_deterministic() -> None:
    model = hierarchical_logistic_model(
        dimension=8,
        num_groups=4,
        observations_per_group=3,
        num_features=2,
        data_seed=17,
    )

    positions = model.initial_position(key=23, num_chains=3)
    repeated = model.initial_position(key=23, num_chains=3)

    assert positions == repeated
    assert len(positions) == 3
    assert all(len(position) == 8 for position in positions)


def test_hierarchical_logistic_log_prob_and_gradient_are_finite() -> None:
    model = hierarchical_logistic_model(
        dimension=8,
        num_groups=4,
        observations_per_group=3,
        num_features=2,
        data_seed=17,
    )
    positions = model.initial_position(key=0, num_chains=4)
    log_probs = vectorized_log_prob(model, positions)

    assert len(log_probs) == 4
    assert all(math.isfinite(value) for value in log_probs)

    position = jnp.asarray(positions[0], dtype=jnp.float32)
    value, grad = jax.value_and_grad(lambda q: model.log_prob(q))(position)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_hierarchical_logistic_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="requires dimension 8"):
        HierarchicalLogisticModel(
            dimension=7,
            num_groups=4,
            observations_per_group=3,
            num_features=2,
        )


def test_registry_loads_stochastic_volatility_with_time_metadata() -> None:
    model = get_model(
        "stochastic_volatility",
        dimension=6,
        model_config={
            "time_length": 6,
            "persistence": 0.9,
            "innovation_scale": 0.25,
            "data_seed": 29,
        },
    )
    metadata = model.metadata.as_dict()

    assert "stochastic_volatility" in available_models()
    assert model.name == "stochastic_volatility"
    assert model.dimension == 6
    assert metadata["event_shape"] == [6]
    assert metadata["model_family"] == "stochastic_volatility"
    assert metadata["time_length"] == 6
    assert metadata["num_observations"] == 6


def test_stochastic_volatility_synthetic_data_is_deterministic() -> None:
    data = generate_synthetic_stochastic_volatility_data(
        time_length=7,
        persistence=0.9,
        innovation_scale=0.25,
        seed=31,
    )
    repeated = generate_synthetic_stochastic_volatility_data(
        time_length=7,
        persistence=0.9,
        innovation_scale=0.25,
        seed=31,
    )

    assert data == repeated
    assert data.time_length == 7
    assert len(data.observations) == 7
    assert len(data.true_log_volatility) == 7


def test_stochastic_volatility_initial_positions_are_deterministic() -> None:
    model = stochastic_volatility_model(
        dimension=6,
        time_length=6,
        persistence=0.9,
        innovation_scale=0.25,
        data_seed=29,
    )

    positions = model.initial_position(key=23, num_chains=3)
    repeated = model.initial_position(key=23, num_chains=3)

    assert positions == repeated
    assert len(positions) == 3
    assert all(len(position) == 6 for position in positions)


def test_stochastic_volatility_log_prob_and_gradient_are_finite() -> None:
    model = stochastic_volatility_model(
        dimension=6,
        time_length=6,
        persistence=0.9,
        innovation_scale=0.25,
        data_seed=29,
    )
    positions = model.initial_position(key=0, num_chains=4)
    log_probs = vectorized_log_prob(model, positions)

    assert len(log_probs) == 4
    assert all(math.isfinite(value) for value in log_probs)

    position = jnp.asarray(positions[0], dtype=jnp.float32)
    value, grad = jax.value_and_grad(lambda q: model.log_prob(q))(position)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_stochastic_volatility_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="requires dimension equal to time_length"):
        StochasticVolatilityModel(dimension=5, time_length=6)


def test_registry_loads_gaussian_process_with_dense_algebra_metadata() -> None:
    model = get_model(
        "gaussian_process",
        dimension=3,
        model_config={
            "num_observations": 6,
            "data_seed": 37,
            "true_amplitude": 1.0,
            "true_length_scale": 0.35,
            "true_noise_scale": 0.08,
            "jitter": 1.0e-5,
        },
    )
    metadata = model.metadata.as_dict()

    assert "gaussian_process" in available_models()
    assert model.name == "gaussian_process"
    assert model.dimension == 3
    assert metadata["event_shape"] == [3]
    assert metadata["model_family"] == "gaussian_process"
    assert metadata["parameterization"] == "log_hyperparameters"
    assert metadata["kernel"] == "squared_exponential"
    assert metadata["input_dimension"] == 1
    assert metadata["num_observations"] == 6


def test_gaussian_process_synthetic_data_is_deterministic() -> None:
    data = generate_synthetic_gaussian_process_data(
        num_observations=6,
        seed=41,
        true_amplitude=1.0,
        true_length_scale=0.35,
        true_noise_scale=0.08,
        jitter=1.0e-5,
    )
    repeated = generate_synthetic_gaussian_process_data(
        num_observations=6,
        seed=41,
        true_amplitude=1.0,
        true_length_scale=0.35,
        true_noise_scale=0.08,
        jitter=1.0e-5,
    )

    assert data == repeated
    assert data.num_observations == 6
    assert len(data.inputs) == 6
    assert len(data.observations) == 6
    assert len(data.true_latent_function) == 6


def test_gaussian_process_initial_positions_are_deterministic() -> None:
    model = gaussian_process_model(
        dimension=3,
        num_observations=6,
        data_seed=37,
    )

    positions = model.initial_position(key=23, num_chains=3)
    repeated = model.initial_position(key=23, num_chains=3)

    assert positions == repeated
    assert len(positions) == 3
    assert all(len(position) == 3 for position in positions)


def test_gaussian_process_log_prob_and_gradient_are_finite() -> None:
    model = gaussian_process_model(
        dimension=3,
        num_observations=6,
        data_seed=37,
        jitter=1.0e-5,
    )
    positions = model.initial_position(key=0, num_chains=4)
    log_probs = vectorized_log_prob(model, positions)

    assert len(log_probs) == 4
    assert all(math.isfinite(value) for value in log_probs)

    position = jnp.asarray(positions[0], dtype=jnp.float32)
    value, grad = jax.value_and_grad(lambda q: model.log_prob(q))(position)
    assert bool(jnp.isfinite(value))
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_gaussian_process_cholesky_failure_has_clear_message() -> None:
    data = GaussianProcessData(
        inputs=(0.0, 0.0),
        observations=(1.0, 1.0),
        true_latent_function=(1.0, 1.0),
        true_amplitude=1.0,
        true_length_scale=1.0,
        true_noise_scale=0.1,
    )
    model = GaussianProcessModel(
        dimension=3,
        num_observations=2,
        jitter=0.0,
        synthetic_data=data,
    )

    with pytest.raises(ValueError, match="GP covariance Cholesky failed"):
        model.log_prob([0.0, 0.0, -400.0])


def test_gaussian_process_rejects_wrong_dimension() -> None:
    with pytest.raises(ValueError, match="requires dimension 3"):
        GaussianProcessModel(dimension=4)
