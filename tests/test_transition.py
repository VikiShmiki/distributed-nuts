from __future__ import annotations

import math

import jax.numpy as jnp
import jax.random as jr

from abnuts.models.funnel import FunnelModel
from abnuts.nuts.transition import new_sampler_state, one_chain_nuts_transition


def test_one_chain_nuts_transition_runs_on_funnel_with_metrics() -> None:
    model = FunnelModel(dimension=4)
    position = jnp.asarray(model.initial_position(key=29, num_chains=1)[0], dtype=jnp.float32)
    state = new_sampler_state(model, position)

    next_state, info, next_key = one_chain_nuts_transition(
        model,
        state,
        jr.PRNGKey(3),
        step_size=0.05,
        max_tree_depth=3,
    )

    assert next_state.position.shape == position.shape
    assert next_key.shape == (2,)
    assert jnp.all(jnp.isfinite(next_state.position))
    assert math.isfinite(float(next_state.potential_energy))
    assert jnp.all(jnp.isfinite(next_state.potential_energy_grad))
    assert 0.0 <= float(info.acceptance_statistic) <= 1.0
    assert 1 <= int(info.realized_tree_depth) <= 3
    assert int(info.leapfrog_count) >= 1
    assert math.isfinite(float(info.energy_error))
    assert math.isfinite(float(info.gradient_norm))
    assert bool(info.divergence_flag) is False


def test_one_chain_nuts_transition_is_deterministic_for_same_state_and_key() -> None:
    model = FunnelModel(dimension=5)
    position = jnp.asarray(model.initial_position(key=7, num_chains=1)[0], dtype=jnp.float32)
    state = new_sampler_state(model, position)
    key = jr.PRNGKey(13)

    first_state, first_info, first_next_key = one_chain_nuts_transition(
        model,
        state,
        key,
        step_size=0.04,
        max_tree_depth=4,
    )
    second_state, second_info, second_next_key = one_chain_nuts_transition(
        model,
        state,
        key,
        step_size=0.04,
        max_tree_depth=4,
    )

    assert jnp.array_equal(first_state.position, second_state.position)
    assert jnp.array_equal(
        first_state.potential_energy,
        second_state.potential_energy,
    )
    assert jnp.array_equal(
        first_state.potential_energy_grad,
        second_state.potential_energy_grad,
    )
    assert jnp.array_equal(first_next_key, second_next_key)
    for first_metric, second_metric in zip(first_info, second_info, strict=True):
        assert jnp.array_equal(first_metric, second_metric)
