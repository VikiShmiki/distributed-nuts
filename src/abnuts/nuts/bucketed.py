"""Bucketed multi-chain NUTS runner built around the shared transition."""

from __future__ import annotations

import time
from collections.abc import Sequence
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array, lax

from abnuts.blocking import block_until_ready_tree
from abnuts.models.base import BenchmarkModel
from abnuts.nuts.monolithic import monolithic_transition, new_multi_chain_state
from abnuts.nuts.planner import BucketPlan, make_bucket_plan
from abnuts.nuts.predictors import (
    PredictorState,
    hvp_curvature_work,
    new_predictor_state,
    predict_work,
    predictor_uses_hvp,
    update_hvp_work,
    update_predictor_state,
)
from abnuts.nuts.state import SamplerState
from abnuts.nuts.transition import TransitionInfo
from abnuts.profiling import TimingBreakdown, TimingRecorder


class BucketedRunResult(NamedTuple):
    """Outputs from a fixed-step bucketed multi-chain run."""

    initial_state: SamplerState
    final_state: SamplerState
    initial_rng_keys: Array
    final_rng_keys: Array
    trace_positions: Array
    transition_info: TransitionInfo
    predictor_state: PredictorState
    bucket_plans: tuple[BucketPlan, ...]
    hvp_overhead_seconds: float
    hvp_call_count: int
    timing: TimingBreakdown = TimingBreakdown()


def bucketed_transition(
    model: BenchmarkModel,
    state: SamplerState,
    rng_keys: Array,
    plan: BucketPlan,
    *,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
    timing_recorder: TimingRecorder | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Apply one NUTS transition per real chain according to a bucket plan.

    Padded lanes are executed with repeated valid chain indices so the bucket has
    a fixed shape, but only true entries in ``plan.mask`` are scattered back.
    """
    if max_tree_depth <= 0:
        raise ValueError(f"max_tree_depth must be positive, got {max_tree_depth!r}")

    num_chains = state.position.shape[0]
    if rng_keys.shape != (num_chains, 2):
        raise ValueError(
            "rng_keys must have shape (num_chains, 2); "
            f"got {rng_keys.shape} for {num_chains} chains"
        )

    if timing_recorder is not None and timing_recorder.enabled:
        (bucket_state, bucket_keys), _ = timing_recorder.timed_call(
            "gather",
            partial(_jit_gather_bucket_rectangle_inputs, state, rng_keys, plan.idx),
            block_before=(state, rng_keys, plan.idx),
            marker_name="abnuts:bucket_gather",
        )
        (bucket_next_state, bucket_info, bucket_next_keys), _ = (
            timing_recorder.timed_call(
                "executor",
                partial(
                    _jit_fixed_shape_bucket_executor,
                    model,
                    bucket_state,
                    bucket_keys,
                    step_size=step_size,
                    max_tree_depth=max_tree_depth,
                    inverse_mass_matrix=inverse_mass_matrix,
                    divergence_threshold=divergence_threshold,
                    data=data,
                ),
                block_before=(bucket_state, bucket_keys),
                marker_name="abnuts:bucket_executor",
            )
        )
        (next_state, next_info, next_rng_keys), _ = timing_recorder.timed_call(
            "scatter",
            partial(
                _jit_scatter_bucket_rectangle_outputs,
                state,
                plan.idx,
                plan.mask,
                bucket_next_state,
                bucket_info,
                bucket_next_keys,
            ),
            block_before=(bucket_next_state, bucket_info, bucket_next_keys),
            marker_name="abnuts:bucket_scatter",
        )
        return next_state, next_info, next_rng_keys

    return _jit_fixed_shape_bucketed_transition(
        model,
        state,
        rng_keys,
        plan.idx,
        plan.mask,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
        inverse_mass_matrix=inverse_mass_matrix,
        divergence_threshold=divergence_threshold,
        data=data,
    )


def run_bucketed(
    model: BenchmarkModel,
    initial_positions: Array,
    rng_key: Array,
    *,
    num_steps: int,
    step_size: float | Array,
    max_tree_depth: int,
    bucket_size: int | Sequence[int],
    predictor: str = "history",
    predictor_beta: float = 0.9,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    dtype: Any = jnp.float32,
    data: Any | None = None,
    enable_timing_breakdown: bool = False,
    enable_profiler_markers: bool = False,
) -> BucketedRunResult:
    """Run a tiny fixed-step bucketed multi-chain NUTS loop."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps!r}")

    positions = jnp.asarray(initial_positions, dtype=dtype)
    state = new_multi_chain_state(model, positions, data=data)
    initial_state = state
    initial_rng_keys = jr.split(rng_key, positions.shape[0])
    rng_keys = initial_rng_keys
    predictor_state = new_predictor_state(positions.shape[0], dtype=dtype)
    predictor_key = rng_key

    trace_positions: list[Array] = []
    infos: list[TransitionInfo] = []
    plans: list[BucketPlan] = []
    hvp_overhead_seconds = 0.0
    hvp_call_count = 0
    timing_recorder = TimingRecorder(
        enabled=enable_timing_breakdown,
        enable_profiler_markers=enable_profiler_markers,
    )
    for _ in range(num_steps):
        predictor_key, prediction_key = jr.split(predictor_key)
        if predictor_uses_hvp(predictor):
            if timing_recorder.enabled:
                hvp_work, hvp_elapsed = timing_recorder.timed_call(
                    "hvp",
                    partial(hvp_curvature_work, model, state.position, data=data),
                    block_before=state,
                    marker_name="abnuts:hvp_probe",
                )
                hvp_overhead_seconds += hvp_elapsed
            else:
                state = block_until_ready_tree(state)
                hvp_start = time.perf_counter()
                hvp_work = hvp_curvature_work(model, state.position, data=data)
                hvp_work = block_until_ready_tree(hvp_work)
                hvp_overhead_seconds += time.perf_counter() - hvp_start
            hvp_call_count += 1
            predictor_state = update_hvp_work(predictor_state, hvp_work)

        (_predicted_work, plan), _ = timing_recorder.timed_call(
            "planner",
            partial(
                _predict_and_plan,
                predictor,
                predictor_state,
                prediction_key=prediction_key,
                bucket_size=bucket_size,
            ),
            block_before=predictor_state,
            marker_name="abnuts:bucket_planner",
        )
        state, info, rng_keys = bucketed_transition(
            model,
            state,
            rng_keys,
            plan,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
            timing_recorder=timing_recorder,
        )
        predictor_state = update_predictor_state(
            predictor_state,
            info.realized_tree_depth.astype(dtype),
            beta=predictor_beta,
        )
        trace_positions.append(state.position)
        infos.append(info)
        plans.append(plan)

    stacked_info = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *infos)
    return BucketedRunResult(
        initial_state=initial_state,
        final_state=state,
        initial_rng_keys=initial_rng_keys,
        final_rng_keys=rng_keys,
        trace_positions=jnp.stack(trace_positions),
        transition_info=stacked_info,
        predictor_state=predictor_state,
        bucket_plans=tuple(plans),
        hvp_overhead_seconds=hvp_overhead_seconds,
        hvp_call_count=hvp_call_count,
        timing=timing_recorder.snapshot(),
    )


def _predict_and_plan(
    predictor: str,
    predictor_state: PredictorState,
    *,
    prediction_key: Array,
    bucket_size: int | Sequence[int],
) -> tuple[Array, BucketPlan]:
    predicted_work = predict_work(
        predictor,
        predictor_state,
        rng_key=prediction_key if predictor == "random" else None,
    )
    plan = make_bucket_plan(
        predicted_work,
        canonical_bucket_sizes=bucket_size,
    )
    return predicted_work, plan


def _fixed_shape_bucketed_transition(
    model: BenchmarkModel,
    state: SamplerState,
    rng_keys: Array,
    bucket_idx: Array,
    bucket_mask: Array,
    *,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Run one rectangular fixed-shape bucket transition in a single JIT program."""
    bucket_state, bucket_keys = _gather_bucket_rectangle_inputs(
        state,
        rng_keys,
        bucket_idx,
    )
    bucket_next_state, bucket_info, bucket_next_keys = _fixed_shape_bucket_executor(
        model,
        bucket_state,
        bucket_keys,
        step_size=step_size,
        max_tree_depth=max_tree_depth,
        inverse_mass_matrix=inverse_mass_matrix,
        divergence_threshold=divergence_threshold,
        data=data,
    )
    return _scatter_bucket_rectangle_outputs(
        state,
        bucket_idx,
        bucket_mask,
        bucket_next_state,
        bucket_info,
        bucket_next_keys,
    )


def _gather_bucket_rectangle_inputs(
    state: SamplerState,
    rng_keys: Array,
    bucket_idx: Array,
) -> tuple[SamplerState, Array]:
    """Gather the full rectangular bucket plan, including padded lanes."""
    return _gather_sampler_state(state, bucket_idx), rng_keys[bucket_idx]


def _fixed_shape_bucket_executor(
    model: BenchmarkModel,
    bucket_state: SamplerState,
    bucket_keys: Array,
    *,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Execute each bucket as its own fixed-shape transition.

    ``lax.map`` scans the bucket axis, so every bucket runs its own vmapped
    transition and therefore its own ``lax.while_loop``. A bucket stops as soon
    as its own deepest chain stops instead of waiting for the deepest chain in
    the whole batch, and that per-bucket exit is the entire mechanism bucketing
    exists to exploit.

    Flattening the rectangle into a single transition would merge every bucket
    back under one ``any(active)`` predicate and hand the reclaimed work
    straight back. That is what the pre-T40 executor did; see the T39/T40 notes
    in ``STATUS.md``. For the same reason this must not become a ``vmap`` over
    the bucket axis, which would also collapse the per-bucket loops into one.

    Scanning is sequential across buckets. That is a real trade-off against the
    single wide ``vmap`` the monolithic baseline gets, and it is the cost the
    reclaimed straggler work has to pay for.
    """

    def run_bucket(
        bucket: tuple[SamplerState, Array],
    ) -> tuple[SamplerState, TransitionInfo, Array]:
        lane_state, lane_keys = bucket
        return monolithic_transition(
            model,
            lane_state,
            lane_keys,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
        )

    return lax.map(run_bucket, (bucket_state, bucket_keys))


def _scatter_bucket_rectangle_outputs(
    reference_state: SamplerState,
    bucket_idx: Array,
    bucket_mask: Array,
    bucket_next_state: SamplerState,
    bucket_info: TransitionInfo,
    bucket_next_keys: Array,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Scatter rectangular bucket outputs, discarding every padded lane."""
    flat_idx = jnp.reshape(bucket_idx, (-1,))
    flat_mask = jnp.reshape(bucket_mask, (-1,))
    flat_next_state = _flatten_bucket_tree(bucket_next_state)
    flat_info = _flatten_bucket_tree(bucket_info)
    flat_next_keys = jnp.reshape(bucket_next_keys, (-1, 2))

    next_state = _scatter_flat_sampler_state(
        _empty_sampler_state_like(reference_state),
        flat_idx,
        flat_mask,
        flat_next_state,
    )
    next_info = _scatter_flat_transition_info(
        _empty_transition_info(
            reference_state.position.shape[0],
            reference_state.position.dtype,
        ),
        flat_idx,
        flat_mask,
        flat_info,
    )
    next_rng_keys = _scatter_masked_flat_array(
        jnp.zeros((reference_state.position.shape[0], 2), dtype=bucket_next_keys.dtype),
        flat_idx,
        flat_mask,
        flat_next_keys,
    )
    return next_state, next_info, next_rng_keys


def _flatten_bucket_tree(tree: Any) -> Any:
    return jax.tree_util.tree_map(_flatten_bucket_array, tree)


def _flatten_bucket_array(array: Array) -> Array:
    return jnp.reshape(array, (array.shape[0] * array.shape[1], *array.shape[2:]))


def _scatter_flat_sampler_state(
    target: SamplerState,
    idx: Array,
    mask: Array,
    values: SamplerState,
) -> SamplerState:
    return SamplerState(
        position=_scatter_masked_flat_array(target.position, idx, mask, values.position),
        potential_energy=_scatter_masked_flat_array(
            target.potential_energy,
            idx,
            mask,
            values.potential_energy,
        ),
        potential_energy_grad=_scatter_masked_flat_array(
            target.potential_energy_grad,
            idx,
            mask,
            values.potential_energy_grad,
        ),
    )


def _scatter_flat_transition_info(
    target: TransitionInfo,
    idx: Array,
    mask: Array,
    values: TransitionInfo,
) -> TransitionInfo:
    return TransitionInfo(
        acceptance_statistic=_scatter_masked_flat_array(
            target.acceptance_statistic,
            idx,
            mask,
            values.acceptance_statistic,
        ),
        divergence_flag=_scatter_masked_flat_array(
            target.divergence_flag,
            idx,
            mask,
            values.divergence_flag,
        ),
        realized_tree_depth=_scatter_masked_flat_array(
            target.realized_tree_depth,
            idx,
            mask,
            values.realized_tree_depth,
        ),
        leapfrog_count=_scatter_masked_flat_array(
            target.leapfrog_count,
            idx,
            mask,
            values.leapfrog_count,
        ),
        energy_error=_scatter_masked_flat_array(
            target.energy_error,
            idx,
            mask,
            values.energy_error,
        ),
        gradient_norm=_scatter_masked_flat_array(
            target.gradient_norm,
            idx,
            mask,
            values.gradient_norm,
        ),
        max_tree_depth_hit=_scatter_masked_flat_array(
            target.max_tree_depth_hit,
            idx,
            mask,
            values.max_tree_depth_hit,
        ),
    )


def _scatter_masked_flat_array(
    target: Array,
    idx: Array,
    mask: Array,
    values: Array,
) -> Array:
    """Scatter real lanes back to their chains, discarding padded lanes.

    Padded lanes are routed to an out-of-bounds index and dropped by the
    scatter itself. The previous implementation appended a sentinel row with
    ``concatenate``, scattered into it, then sliced it off, which cost two full
    array copies on top of the scatter for every field written. With eleven
    fields per transition that overhead was a substantial fraction of the whole
    executor, and it was paid even with a single bucket and no padding at all.
    """
    out_of_bounds = jnp.asarray(target.shape[0], dtype=idx.dtype)
    scatter_idx = jnp.where(mask, idx, out_of_bounds)
    return target.at[scatter_idx].set(values, mode="drop")


def _gather_sampler_state(state: SamplerState, idx: Array) -> SamplerState:
    return SamplerState(
        position=state.position[idx],
        potential_energy=state.potential_energy[idx],
        potential_energy_grad=state.potential_energy_grad[idx],
    )


def _empty_sampler_state_like(state: SamplerState) -> SamplerState:
    return SamplerState(
        position=jnp.zeros_like(state.position),
        potential_energy=jnp.zeros_like(state.potential_energy),
        potential_energy_grad=jnp.zeros_like(state.potential_energy_grad),
    )


def _empty_transition_info(num_chains: int, dtype: Any) -> TransitionInfo:
    return TransitionInfo(
        acceptance_statistic=jnp.zeros((num_chains,), dtype=dtype),
        divergence_flag=jnp.zeros((num_chains,), dtype=jnp.bool_),
        realized_tree_depth=jnp.zeros((num_chains,), dtype=jnp.int32),
        leapfrog_count=jnp.zeros((num_chains,), dtype=jnp.int32),
        energy_error=jnp.zeros((num_chains,), dtype=dtype),
        gradient_norm=jnp.zeros((num_chains,), dtype=dtype),
        max_tree_depth_hit=jnp.zeros((num_chains,), dtype=jnp.bool_),
    )


_jit_gather_bucket_rectangle_inputs = jax.jit(_gather_bucket_rectangle_inputs)
_jit_fixed_shape_bucket_executor = jax.jit(
    _fixed_shape_bucket_executor,
    static_argnames=("model", "max_tree_depth"),
)
_jit_scatter_bucket_rectangle_outputs = jax.jit(_scatter_bucket_rectangle_outputs)
_jit_fixed_shape_bucketed_transition = jax.jit(
    _fixed_shape_bucketed_transition,
    static_argnames=("model", "max_tree_depth"),
)
