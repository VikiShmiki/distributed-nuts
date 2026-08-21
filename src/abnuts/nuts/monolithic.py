"""Vectorized monolithic multi-chain NUTS runner."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from abnuts.models.base import BenchmarkModel
from abnuts.nuts.state import SamplerState
from abnuts.nuts.transition import (
    TransitionInfo,
    new_sampler_state,
    one_chain_nuts_transition,
)


class MonolithicRunResult(NamedTuple):
    """Outputs from a fixed-step monolithic multi-chain run."""

    initial_state: SamplerState
    final_state: SamplerState
    initial_rng_keys: Array
    final_rng_keys: Array
    trace_positions: Array
    transition_info: TransitionInfo


def new_multi_chain_state(
    model: BenchmarkModel,
    positions: Array,
    data: Any | None = None,
) -> SamplerState:
    """Construct a batched sampler state from ``(chains, dimension)`` positions."""
    positions_array = jnp.asarray(positions)
    if positions_array.ndim != 2:
        raise ValueError(
            "positions must have shape (num_chains, dimension); "
            f"got {positions_array.shape}"
        )
    if positions_array.shape[0] <= 0:
        raise ValueError("positions must contain at least one chain")

    def build_state(position: Array) -> SamplerState:
        return new_sampler_state(model, position, data=data)

    return jax.vmap(build_state)(positions_array)


def monolithic_transition(
    model: BenchmarkModel,
    state: SamplerState,
    rng_keys: Array,
    *,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Apply the shared one-chain NUTS transition independently to each chain."""
    if max_tree_depth <= 0:
        raise ValueError(f"max_tree_depth must be positive, got {max_tree_depth!r}")

    position_count = state.position.shape[0]
    if rng_keys.shape != (position_count, 2):
        raise ValueError(
            "rng_keys must have shape (num_chains, 2); "
            f"got {rng_keys.shape} for {position_count} chains"
        )

    def transition_chain(
        chain_state: SamplerState,
        chain_key: Array,
    ) -> tuple[SamplerState, TransitionInfo, Array]:
        return one_chain_nuts_transition(
            model,
            chain_state,
            chain_key,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
        )

    return jax.vmap(transition_chain)(state, rng_keys)


def run_monolithic(
    model: BenchmarkModel,
    initial_positions: Array,
    rng_key: Array,
    *,
    num_steps: int,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    dtype: Any = jnp.float32,
    data: Any | None = None,
) -> MonolithicRunResult:
    """Run a tiny fixed-step monolithic multi-chain NUTS loop."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps!r}")

    positions = jnp.asarray(initial_positions, dtype=dtype)
    state = new_multi_chain_state(model, positions, data=data)
    initial_state = state
    initial_rng_keys = jr.split(rng_key, positions.shape[0])
    rng_keys = initial_rng_keys

    trace_positions: list[Array] = []
    infos: list[TransitionInfo] = []
    for _ in range(num_steps):
        state, info, rng_keys = monolithic_transition(
            model,
            state,
            rng_keys,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
        )
        trace_positions.append(state.position)
        infos.append(info)

    stacked_info = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *infos)
    return MonolithicRunResult(
        initial_state=initial_state,
        final_state=state,
        initial_rng_keys=initial_rng_keys,
        final_rng_keys=rng_keys,
        trace_positions=jnp.stack(trace_positions),
        transition_info=stacked_info,
    )


# Module-level JIT-compiled monolithic transition.
# model and max_tree_depth control the trace structure: model provides log_prob
# (traced symbolically at compile time) and max_tree_depth determines loop
# unrolling depth. Compiled once per (model, max_tree_depth, arg shapes/dtypes)
# combination and reused across warm iterations. Callers must pass the same
# model instance (or an equal frozen-dataclass equivalent) to hit the cache.
jit_monolithic_transition = jax.jit(
    monolithic_transition,
    static_argnames=("model", "max_tree_depth"),
)


def run_monolithic_jit(
    model: BenchmarkModel,
    initial_positions: Array,
    rng_key: Array,
    *,
    num_steps: int,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    dtype: Any = jnp.float32,
    data: Any | None = None,
) -> MonolithicRunResult:
    """Run fixed-step monolithic NUTS using the cached JIT-compiled transition.

    Functionally identical to run_monolithic but uses jit_monolithic_transition,
    ensuring the XLA program is compiled once and reused across all loop iterations.
    The first call triggers compilation; subsequent calls with the same argument
    shapes and the same (model, max_tree_depth) pair reuse the cache.

    Use block_until_ready_tree from abnuts.blocking externally to separate cold
    (includes compile) from warm (cached) wall-clock timings.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps!r}")

    positions = jnp.asarray(initial_positions, dtype=dtype)
    state = new_multi_chain_state(model, positions, data=data)
    initial_state = state
    initial_rng_keys = jr.split(rng_key, positions.shape[0])
    rng_keys = initial_rng_keys

    trace_positions: list[Array] = []
    infos: list[TransitionInfo] = []
    for _ in range(num_steps):
        state, info, rng_keys = jit_monolithic_transition(
            model,
            state,
            rng_keys,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
        )
        trace_positions.append(state.position)
        infos.append(info)

    stacked_info = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *infos)
    return MonolithicRunResult(
        initial_state=initial_state,
        final_state=state,
        initial_rng_keys=initial_rng_keys,
        final_rng_keys=rng_keys,
        trace_positions=jnp.stack(trace_positions),
        transition_info=stacked_info,
    )
