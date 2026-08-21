from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from abnuts.models.funnel import FunnelModel
from abnuts.nuts.monolithic import (
    jit_monolithic_transition,
    monolithic_transition,
    new_multi_chain_state,
    run_monolithic,
    run_monolithic_jit,
)


def test_monolithic_transition_vectorizes_one_chain_transition() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=17, num_chains=8), dtype=jnp.float32)
    state = new_multi_chain_state(model, positions)
    rng_keys = jr.split(jr.PRNGKey(23), positions.shape[0])

    next_state, info, next_keys = monolithic_transition(
        model,
        state,
        rng_keys,
        step_size=0.04,
        max_tree_depth=3,
    )

    assert next_state.position.shape == positions.shape
    assert next_state.potential_energy.shape == (positions.shape[0],)
    assert next_state.potential_energy_grad.shape == positions.shape
    assert next_keys.shape == rng_keys.shape
    assert info.acceptance_statistic.shape == (positions.shape[0],)
    assert info.realized_tree_depth.shape == (positions.shape[0],)
    assert jnp.all(jnp.isfinite(next_state.position))
    assert jnp.all((info.acceptance_statistic >= 0.0) & (info.acceptance_statistic <= 1.0))
    assert jnp.all(info.realized_tree_depth >= 1)
    assert jnp.all(info.realized_tree_depth <= 3)
    assert jnp.all(info.leapfrog_count >= 1)


def test_run_monolithic_records_trace_and_metrics() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=5, num_chains=8), dtype=jnp.float32)

    result = run_monolithic(
        model,
        positions,
        jr.PRNGKey(31),
        num_steps=3,
        step_size=0.03,
        max_tree_depth=3,
    )

    assert result.initial_state.position.shape == positions.shape
    assert result.final_state.position.shape == positions.shape
    assert result.initial_rng_keys.shape == (positions.shape[0], 2)
    assert result.final_rng_keys.shape == (positions.shape[0], 2)
    assert result.trace_positions.shape == (3, positions.shape[0], positions.shape[1])
    assert result.transition_info.acceptance_statistic.shape == (3, positions.shape[0])
    assert result.transition_info.energy_error.shape == (3, positions.shape[0])
    assert jnp.array_equal(result.trace_positions[-1], result.final_state.position)


def test_run_monolithic_is_deterministic_for_same_seed_and_positions() -> None:
    model = FunnelModel(dimension=5)
    positions = jnp.asarray(model.initial_position(key=11, num_chains=6), dtype=jnp.float32)
    key = jr.PRNGKey(41)

    first = run_monolithic(
        model,
        positions,
        key,
        num_steps=2,
        step_size=0.025,
        max_tree_depth=3,
    )
    second = run_monolithic(
        model,
        positions,
        key,
        num_steps=2,
        step_size=0.025,
        max_tree_depth=3,
    )

    assert jnp.array_equal(first.trace_positions, second.trace_positions)
    assert jnp.array_equal(first.final_rng_keys, second.final_rng_keys)
    for first_metric, second_metric in zip(
        first.transition_info,
        second.transition_info,
        strict=True,
    ):
        assert jnp.array_equal(first_metric, second_metric)


def test_jit_monolithic_transition_matches_monolithic_transition() -> None:
    """jit_monolithic_transition must match monolithic_transition within strict tolerance.

    XLA compilation reorders floating-point operations relative to JAX eager mode,
    causing ULP-level differences in float32 results (observed: up to 2 ULP, ~2e-7
    relative). Integer and boolean metrics and RNG keys are bitwise identical.
    Tolerance: atol=1e-5, rtol=1e-5 for float arrays — tighter than any meaningful
    statistical difference.
    """
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=17, num_chains=8), dtype=jnp.float32)
    state = new_multi_chain_state(model, positions)
    rng_keys = jr.split(jr.PRNGKey(23), positions.shape[0])

    ref_state, ref_info, ref_keys = monolithic_transition(
        model, state, rng_keys, step_size=0.04, max_tree_depth=3
    )
    jit_state, jit_info, jit_keys = jit_monolithic_transition(
        model, state, rng_keys, step_size=0.04, max_tree_depth=3
    )

    # Float arrays: tight tolerance because XLA may reorder ops vs eager mode.
    _atol, _rtol = 1e-5, 1e-5
    assert jnp.allclose(jit_state.position, ref_state.position, atol=_atol, rtol=_rtol)
    assert jnp.allclose(jit_state.potential_energy, ref_state.potential_energy, atol=_atol, rtol=_rtol)
    assert jnp.allclose(jit_state.potential_energy_grad, ref_state.potential_energy_grad, atol=_atol, rtol=_rtol)
    # RNG keys are integers — must be bitwise identical (no float arithmetic).
    assert jnp.array_equal(jit_keys, ref_keys)
    # Integer/bool metrics must be bitwise identical; float metrics use tight tolerance.
    assert jnp.array_equal(jit_info.realized_tree_depth, ref_info.realized_tree_depth)
    assert jnp.array_equal(jit_info.leapfrog_count, ref_info.leapfrog_count)
    assert jnp.array_equal(jit_info.divergence_flag, ref_info.divergence_flag)
    assert jnp.array_equal(jit_info.max_tree_depth_hit, ref_info.max_tree_depth_hit)
    assert jnp.allclose(jit_info.acceptance_statistic, ref_info.acceptance_statistic, atol=_atol, rtol=_rtol)
    assert jnp.allclose(jit_info.energy_error, ref_info.energy_error, atol=_atol, rtol=_rtol)
    assert jnp.allclose(jit_info.gradient_norm, ref_info.gradient_norm, atol=_atol, rtol=_rtol)


def test_run_monolithic_jit_matches_run_monolithic() -> None:
    """run_monolithic_jit must match run_monolithic within strict tolerance.

    See test_jit_monolithic_transition_matches_monolithic_transition for why
    bitwise equality is impossible for float32 arrays under JIT vs eager.
    Integer/bool arrays and RNG keys remain bitwise identical.
    """
    model = FunnelModel(dimension=5)
    positions = jnp.asarray(model.initial_position(key=11, num_chains=6), dtype=jnp.float32)
    key = jr.PRNGKey(41)

    ref = run_monolithic(
        model, positions, key, num_steps=3, step_size=0.025, max_tree_depth=3
    )
    jit_result = run_monolithic_jit(
        model, positions, key, num_steps=3, step_size=0.025, max_tree_depth=3
    )

    _atol, _rtol = 1e-5, 1e-5
    assert jnp.allclose(jit_result.trace_positions, ref.trace_positions, atol=_atol, rtol=_rtol)
    assert jnp.allclose(jit_result.final_state.position, ref.final_state.position, atol=_atol, rtol=_rtol)
    # RNG keys must be bitwise identical.
    assert jnp.array_equal(jit_result.final_rng_keys, ref.final_rng_keys)
    assert jnp.array_equal(jit_result.initial_rng_keys, ref.initial_rng_keys)
    # Integer/bool transition metrics must be bitwise identical.
    assert jnp.array_equal(jit_result.transition_info.realized_tree_depth, ref.transition_info.realized_tree_depth)
    assert jnp.array_equal(jit_result.transition_info.leapfrog_count, ref.transition_info.leapfrog_count)
    assert jnp.array_equal(jit_result.transition_info.divergence_flag, ref.transition_info.divergence_flag)
    assert jnp.array_equal(jit_result.transition_info.max_tree_depth_hit, ref.transition_info.max_tree_depth_hit)
    # Float metrics use tight tolerance.
    assert jnp.allclose(jit_result.transition_info.acceptance_statistic, ref.transition_info.acceptance_statistic, atol=_atol, rtol=_rtol)
    assert jnp.allclose(jit_result.transition_info.energy_error, ref.transition_info.energy_error, atol=_atol, rtol=_rtol)


def test_jit_monolithic_transition_is_deterministic_across_calls() -> None:
    """Repeated calls to jit_monolithic_transition with the same inputs must agree."""
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=3, num_chains=4), dtype=jnp.float32)
    state = new_multi_chain_state(model, positions)
    rng_keys = jr.split(jr.PRNGKey(7), positions.shape[0])

    first_state, first_info, first_keys = jit_monolithic_transition(
        model, state, rng_keys, step_size=0.03, max_tree_depth=3
    )
    second_state, second_info, second_keys = jit_monolithic_transition(
        model, state, rng_keys, step_size=0.03, max_tree_depth=3
    )

    assert jnp.array_equal(first_state.position, second_state.position)
    assert jnp.array_equal(first_keys, second_keys)
    for first_metric, second_metric in zip(first_info, second_info, strict=True):
        assert jnp.array_equal(first_metric, second_metric)
