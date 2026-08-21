from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from abnuts.models.funnel import FunnelModel
from abnuts.models.gaussian_process import gaussian_process_model
from abnuts.nuts.bucketed import (
    _scatter_bucket_rectangle_outputs,
    bucketed_transition,
    run_bucketed,
)
from abnuts.nuts.monolithic import monolithic_transition, new_multi_chain_state, run_monolithic
from abnuts.nuts.planner import make_bucket_plan
from abnuts.nuts.state import SamplerState
from abnuts.nuts.transition import TransitionInfo


_FLOAT_ATOL = 1e-5
_FLOAT_RTOL = 1e-5


def _assert_jitted_bucket_state_matches_monolithic(bucket_state, mono_state) -> None:
    """Compare jitted bucketed floats with strict tolerance; discrete fields stay exact."""
    assert jnp.allclose(
        bucket_state.position,
        mono_state.position,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    assert jnp.allclose(
        bucket_state.potential_energy,
        mono_state.potential_energy,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    assert jnp.allclose(
        bucket_state.potential_energy_grad,
        mono_state.potential_energy_grad,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )


def _assert_jitted_bucket_info_matches_monolithic(bucket_info, mono_info) -> None:
    """JIT may reorder float32 ops; integer and boolean transition metrics are exact."""
    assert jnp.allclose(
        bucket_info.acceptance_statistic,
        mono_info.acceptance_statistic,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    assert jnp.allclose(
        bucket_info.energy_error,
        mono_info.energy_error,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    assert jnp.allclose(
        bucket_info.gradient_norm,
        mono_info.gradient_norm,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    assert jnp.array_equal(bucket_info.divergence_flag, mono_info.divergence_flag)
    assert jnp.array_equal(bucket_info.realized_tree_depth, mono_info.realized_tree_depth)
    assert jnp.array_equal(bucket_info.leapfrog_count, mono_info.leapfrog_count)
    assert jnp.array_equal(bucket_info.max_tree_depth_hit, mono_info.max_tree_depth_hit)


def test_bucketed_one_step_matches_monolithic_for_real_chains() -> None:
    """Bucketed fixed-shape JIT preserves monolithic transition metrics.

    The repaired bucket executor is compiled, while monolithic_transition here is
    eager. As in the T33 monolithic-JIT tests, XLA may introduce ULP-level float32
    differences. The strict 1e-5 tolerance applies only to float arrays; RNG keys,
    depths, leapfrog counts, and flags must remain bitwise identical.
    """
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=19, num_chains=8), dtype=jnp.float32)
    state = new_multi_chain_state(model, positions)
    rng_keys = jr.split(jr.PRNGKey(23), positions.shape[0])
    plan = make_bucket_plan(
        jnp.asarray([2.0, 1.0, 3.0, 1.5, 2.5, 0.5, 4.0, 3.5], dtype=jnp.float32),
        canonical_bucket_sizes=4,
    )

    mono_state, mono_info, mono_keys = monolithic_transition(
        model,
        state,
        rng_keys,
        step_size=0.04,
        max_tree_depth=3,
    )
    bucket_state, bucket_info, bucket_keys = bucketed_transition(
        model,
        state,
        rng_keys,
        plan,
        step_size=0.04,
        max_tree_depth=3,
    )

    _assert_jitted_bucket_state_matches_monolithic(bucket_state, mono_state)
    assert jnp.array_equal(bucket_keys, mono_keys)
    _assert_jitted_bucket_info_matches_monolithic(bucket_info, mono_info)


def test_bucketed_one_step_matches_monolithic_for_minimal_gaussian_process() -> None:
    """Regression gate for GP leapfrog mismatches seen in diagnostic run 21070."""
    model = gaussian_process_model(dimension=3, num_observations=8)
    positions = jnp.asarray(model.initial_position(key=101, num_chains=5), dtype=jnp.float32)
    state = new_multi_chain_state(model, positions)
    rng_keys = jr.split(jr.PRNGKey(103), positions.shape[0])
    plan = make_bucket_plan(
        jnp.asarray([2.0, 1.0, 3.0, 1.5, 0.5], dtype=jnp.float32),
        canonical_bucket_sizes=4,
    )

    mono_state, mono_info, mono_keys = monolithic_transition(
        model,
        state,
        rng_keys,
        step_size=0.005,
        max_tree_depth=2,
    )
    bucket_state, bucket_info, bucket_keys = bucketed_transition(
        model,
        state,
        rng_keys,
        plan,
        step_size=0.005,
        max_tree_depth=2,
    )

    assert int(plan.padding_count) > 0
    _assert_jitted_bucket_state_matches_monolithic(bucket_state, mono_state)
    assert jnp.array_equal(bucket_keys, mono_keys)
    _assert_jitted_bucket_info_matches_monolithic(bucket_info, mono_info)


def test_padded_lanes_do_not_alter_real_chain_outputs() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=29, num_chains=5), dtype=jnp.float32)
    state = new_multi_chain_state(model, positions)
    rng_keys = jr.split(jr.PRNGKey(31), positions.shape[0])

    no_padding_plan = make_bucket_plan(jnp.arange(5, dtype=jnp.float32), canonical_bucket_sizes=1)
    padded_plan = make_bucket_plan(jnp.arange(5, dtype=jnp.float32), canonical_bucket_sizes=4)

    no_padding_state, no_padding_info, no_padding_keys = bucketed_transition(
        model,
        state,
        rng_keys,
        no_padding_plan,
        step_size=0.035,
        max_tree_depth=3,
    )
    padded_state, padded_info, padded_keys = bucketed_transition(
        model,
        state,
        rng_keys,
        padded_plan,
        step_size=0.035,
        max_tree_depth=3,
    )

    assert int(padded_plan.padding_count) > 0
    assert jnp.array_equal(padded_state.position, no_padding_state.position)
    assert jnp.array_equal(padded_keys, no_padding_keys)
    for padded_metric, no_padding_metric in zip(padded_info, no_padding_info, strict=True):
        assert jnp.array_equal(padded_metric, no_padding_metric)


def test_padded_lane_scatter_ignores_poisoned_false_mask_values() -> None:
    """False-mask lanes must scatter to the sentinel row, not a real chain."""
    reference_state = SamplerState(
        position=jnp.zeros((3, 2), dtype=jnp.float32),
        potential_energy=jnp.zeros((3,), dtype=jnp.float32),
        potential_energy_grad=jnp.zeros((3, 2), dtype=jnp.float32),
    )
    bucket_idx = jnp.asarray([[0, 1], [2, 0]], dtype=jnp.int32)
    bucket_mask = jnp.asarray([[True, True], [True, False]])
    bucket_next_state = SamplerState(
        position=jnp.asarray(
            [
                [[10.0, 11.0], [20.0, 21.0]],
                [[30.0, 31.0], [999.0, 999.0]],
            ],
            dtype=jnp.float32,
        ),
        potential_energy=jnp.asarray([[1.0, 2.0], [3.0, 999.0]], dtype=jnp.float32),
        potential_energy_grad=jnp.asarray(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [999.0, 999.0]],
            ],
            dtype=jnp.float32,
        ),
    )
    bucket_info = TransitionInfo(
        acceptance_statistic=jnp.asarray([[0.1, 0.2], [0.3, 0.99]], dtype=jnp.float32),
        divergence_flag=jnp.asarray([[False, True], [False, True]]),
        realized_tree_depth=jnp.asarray([[1, 2], [3, 99]], dtype=jnp.int32),
        leapfrog_count=jnp.asarray([[1, 3], [7, 99]], dtype=jnp.int32),
        energy_error=jnp.asarray([[0.01, 0.02], [0.03, 9.99]], dtype=jnp.float32),
        gradient_norm=jnp.asarray([[1.1, 1.2], [1.3, 9.9]], dtype=jnp.float32),
        max_tree_depth_hit=jnp.asarray([[False, True], [False, True]]),
    )
    bucket_next_keys = jnp.asarray(
        [
            [[10, 11], [20, 21]],
            [[30, 31], [999, 999]],
        ],
        dtype=jnp.uint32,
    )

    state, info, keys = _scatter_bucket_rectangle_outputs(
        reference_state,
        bucket_idx,
        bucket_mask,
        bucket_next_state,
        bucket_info,
        bucket_next_keys,
    )

    assert jnp.array_equal(
        state.position,
        jnp.asarray([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        state.potential_energy,
        jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        state.potential_energy_grad,
        jnp.asarray([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        info.acceptance_statistic,
        jnp.asarray([0.1, 0.2, 0.3], dtype=jnp.float32),
    )
    assert jnp.array_equal(info.divergence_flag, jnp.asarray([False, True, False]))
    assert jnp.array_equal(info.realized_tree_depth, jnp.asarray([1, 2, 3], dtype=jnp.int32))
    assert jnp.array_equal(info.leapfrog_count, jnp.asarray([1, 3, 7], dtype=jnp.int32))
    assert jnp.array_equal(
        info.energy_error,
        jnp.asarray([0.01, 0.02, 0.03], dtype=jnp.float32),
    )
    assert jnp.array_equal(
        info.gradient_norm,
        jnp.asarray([1.1, 1.2, 1.3], dtype=jnp.float32),
    )
    assert jnp.array_equal(info.max_tree_depth_hit, jnp.asarray([False, True, False]))
    assert jnp.array_equal(
        keys,
        jnp.asarray([[10, 11], [20, 21], [30, 31]], dtype=jnp.uint32),
    )


def test_bucketed_run_matches_one_step_monolithic_run_with_history_predictor() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=7, num_chains=8), dtype=jnp.float32)
    key = jr.PRNGKey(11)

    mono = run_monolithic(
        model,
        positions,
        key,
        num_steps=1,
        step_size=0.03,
        max_tree_depth=3,
    )
    bucket = run_bucketed(
        model,
        positions,
        key,
        num_steps=1,
        step_size=0.03,
        max_tree_depth=3,
        bucket_size=4,
        predictor="history",
    )

    assert jnp.allclose(
        bucket.trace_positions,
        mono.trace_positions,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    assert jnp.array_equal(bucket.final_rng_keys, mono.final_rng_keys)
    _assert_jitted_bucket_info_matches_monolithic(
        bucket.transition_info,
        mono.transition_info,
    )


def test_bucketed_run_matches_one_step_monolithic_run_with_hybrid_predictor() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=37, num_chains=8), dtype=jnp.float32)
    key = jr.PRNGKey(43)

    mono = run_monolithic(
        model,
        positions,
        key,
        num_steps=1,
        step_size=0.03,
        max_tree_depth=3,
    )
    bucket = run_bucketed(
        model,
        positions,
        key,
        num_steps=1,
        step_size=0.03,
        max_tree_depth=3,
        bucket_size=4,
        predictor="hybrid",
    )

    assert bucket.hvp_call_count == 1
    assert bucket.hvp_overhead_seconds >= 0.0
    assert jnp.allclose(
        bucket.trace_positions,
        mono.trace_positions,
        atol=_FLOAT_ATOL,
        rtol=_FLOAT_RTOL,
    )
    assert jnp.array_equal(bucket.final_rng_keys, mono.final_rng_keys)
    _assert_jitted_bucket_info_matches_monolithic(
        bucket.transition_info,
        mono.transition_info,
    )
