from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from abnuts.models.funnel import FunnelModel
from abnuts.nuts.predictors import (
    hvp_curvature_work,
    new_predictor_state,
    predict_work,
    predictor_uses_hvp,
    update_hvp_work,
    update_predictor_state,
)


def test_predictor_outputs_have_chain_shape() -> None:
    state = new_predictor_state(num_chains=5, initial_work=2.0)

    none_prediction = predict_work("none", state)
    history_prediction = predict_work("history", state)
    random_prediction = predict_work("random", state, rng_key=jr.PRNGKey(9))
    hvp_prediction = predict_work("hvp", state)
    hybrid_prediction = predict_work("hybrid", state)

    assert none_prediction.shape == (5,)
    assert history_prediction.shape == (5,)
    assert random_prediction.shape == (5,)
    assert hvp_prediction.shape == (5,)
    assert hybrid_prediction.shape == (5,)
    assert jnp.array_equal(none_prediction, jnp.ones((5,), dtype=jnp.float32))
    assert jnp.array_equal(history_prediction, jnp.full((5,), 2.0, dtype=jnp.float32))
    assert jnp.array_equal(hvp_prediction, jnp.full((5,), 2.0, dtype=jnp.float32))
    assert jnp.array_equal(hybrid_prediction, jnp.full((5,), 2.0, dtype=jnp.float32))


def test_last_depth_predictor_is_causal_and_uses_unit_initial_scores() -> None:
    state = new_predictor_state(num_chains=3, initial_work=7.0)
    np.testing.assert_array_equal(predict_work("last_depth", state), np.ones(3))

    realized = jnp.asarray([2.0, 5.0, 3.0])
    updated = update_predictor_state(state, realized, beta=0.9)
    np.testing.assert_array_equal(predict_work("last_depth", updated), realized)


def test_history_predictor_moves_toward_realized_depth() -> None:
    state = new_predictor_state(num_chains=4, initial_work=1.0)
    realized_depth = jnp.asarray([3.0, 5.0, 7.0, 9.0], dtype=jnp.float32)

    updated = update_predictor_state(state, realized_depth, beta=0.5)

    assert jnp.allclose(updated.history_ema_work, jnp.asarray([2.0, 3.0, 4.0, 5.0]))
    assert jnp.array_equal(updated.last_realized_work, realized_depth)
    assert int(updated.num_updates) == 1
    assert jnp.array_equal(predict_work("history", updated), updated.history_ema_work)


def test_hvp_curvature_work_has_chain_shape_and_is_deterministic() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=41, num_chains=3), dtype=jnp.float32)

    first = hvp_curvature_work(model, positions)
    second = hvp_curvature_work(model, positions)

    assert first.shape == (3,)
    assert jnp.all(jnp.isfinite(first))
    assert jnp.array_equal(first, second)
    assert jnp.all(first >= 1.0)


def test_hybrid_predictor_uses_conservative_work_maximum() -> None:
    state = new_predictor_state(
        num_chains=4,
        initial_work=jnp.asarray([1.0, 5.0, 3.0, 2.0], dtype=jnp.float32),
    )
    state = update_hvp_work(
        state,
        jnp.asarray([4.0, 2.0, 3.5, 1.0], dtype=jnp.float32),
    )

    assert predictor_uses_hvp("hvp")
    assert predictor_uses_hvp("hybrid")
    assert not predictor_uses_hvp("history")
    assert jnp.array_equal(predict_work("hvp", state), state.hvp_work)
    assert jnp.array_equal(
        predict_work("hybrid", state),
        jnp.asarray([4.0, 5.0, 3.5, 2.0], dtype=jnp.float32),
    )


def test_random_predictor_is_deterministic_under_fixed_key() -> None:
    state = new_predictor_state(num_chains=6)
    key = jr.PRNGKey(17)

    first = predict_work("random", state, rng_key=key)
    second = predict_work("random", state, rng_key=key)

    assert jnp.array_equal(first, second)


def test_predictor_validation_errors_are_informative() -> None:
    state = new_predictor_state(num_chains=3)

    with pytest.raises(ValueError, match="random predictor requires rng_key"):
        predict_work("random", state)
    with pytest.raises(ValueError, match="unknown predictor mode"):
        predict_work("oracle_current", state)
    with pytest.raises(ValueError, match="shape"):
        update_hvp_work(state, jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match="shape"):
        update_predictor_state(state, jnp.asarray([1.0, 2.0]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        update_predictor_state(state, jnp.ones((3,)), beta=1.5)
