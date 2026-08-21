"""Local independent-chain NUTS baseline.

This is a same-process baseline that executes each chain separately while reusing
the shared one-chain NUTS transition. It is intentionally labeled as local, not
as a process-parallel CPU implementation.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from abnuts.models.base import BenchmarkModel
from abnuts.nuts.monolithic import new_multi_chain_state
from abnuts.nuts.state import SamplerState
from abnuts.nuts.transition import TransitionInfo, one_chain_nuts_transition


class IndependentChainRunResult(NamedTuple):
    """Outputs from a same-process independent-chain NUTS baseline run."""

    initial_state: SamplerState
    final_state: SamplerState
    initial_rng_keys: Array
    final_rng_keys: Array
    trace_positions: Array
    transition_info: TransitionInfo
    baseline_type: str


def independent_chain_transition(
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
    """Apply NUTS to each chain sequentially in the host process."""
    num_chains = state.position.shape[0]
    if rng_keys.shape != (num_chains, 2):
        raise ValueError(
            "rng_keys must have shape (num_chains, 2); "
            f"got {rng_keys.shape} for {num_chains} chains"
        )

    next_states: list[SamplerState] = []
    next_infos: list[TransitionInfo] = []
    next_keys: list[Array] = []
    for chain_index in range(num_chains):
        chain_state = SamplerState(
            position=state.position[chain_index],
            potential_energy=state.potential_energy[chain_index],
            potential_energy_grad=state.potential_energy_grad[chain_index],
        )
        next_state, info, next_key = one_chain_nuts_transition(
            model,
            chain_state,
            rng_keys[chain_index],
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
        )
        next_states.append(next_state)
        next_infos.append(info)
        next_keys.append(next_key)

    return (
        _stack_sampler_states(next_states),
        jax.tree_util.tree_map(lambda *items: jnp.stack(items), *next_infos),
        jnp.stack(next_keys),
    )


def run_independent_chains_local(
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
) -> IndependentChainRunResult:
    """Run NUTS chains independently in the local Python process."""
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
        state, info, rng_keys = independent_chain_transition(
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
    return IndependentChainRunResult(
        initial_state=initial_state,
        final_state=state,
        initial_rng_keys=initial_rng_keys,
        final_rng_keys=rng_keys,
        trace_positions=jnp.stack(trace_positions),
        transition_info=stacked_info,
        baseline_type="independent_chain_local",
    )


def _stack_sampler_states(states: list[SamplerState]) -> SamplerState:
    return SamplerState(
        position=jnp.stack([state.position for state in states]),
        potential_energy=jnp.stack([state.potential_energy for state in states]),
        potential_energy_grad=jnp.stack(
            [state.potential_energy_grad for state in states]
        ),
    )
