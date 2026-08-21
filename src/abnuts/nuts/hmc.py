"""Fixed-length HMC baseline built from the shared leapfrog integrator."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from abnuts.models.base import BenchmarkModel
from abnuts.nuts.hamiltonian import hamiltonian_energy
from abnuts.nuts.integrator import leapfrog_step
from abnuts.nuts.monolithic import new_multi_chain_state
from abnuts.nuts.state import IntegratorState, SamplerState
from abnuts.nuts.transition import TransitionInfo


class FixedHmcRunResult(NamedTuple):
    """Outputs from a fixed-length HMC multi-chain baseline run."""

    initial_state: SamplerState
    final_state: SamplerState
    initial_rng_keys: Array
    final_rng_keys: Array
    trace_positions: Array
    transition_info: TransitionInfo
    num_leapfrog_steps: int


def one_chain_fixed_hmc_transition(
    model: BenchmarkModel,
    state: SamplerState,
    rng_key: Array,
    *,
    step_size: float | Array,
    num_leapfrog_steps: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Run one fixed-length HMC transition for a single chain."""
    if num_leapfrog_steps <= 0:
        raise ValueError(
            f"num_leapfrog_steps must be positive, got {num_leapfrog_steps!r}"
        )

    key, momentum_key, accept_key = jr.split(rng_key, 3)
    dtype = state.position.dtype
    step = jnp.asarray(step_size, dtype=dtype)
    threshold = jnp.asarray(divergence_threshold, dtype=dtype)
    momentum = jr.normal(momentum_key, shape=state.position.shape, dtype=dtype)

    initial = IntegratorState(
        position=state.position,
        momentum=momentum,
        potential_energy=state.potential_energy,
        potential_energy_grad=state.potential_energy_grad,
    )
    initial_energy = hamiltonian_energy(initial, inverse_mass_matrix)
    proposal = initial
    for _ in range(num_leapfrog_steps):
        proposal = leapfrog_step(
            model,
            proposal,
            step_size=step,
            inverse_mass_matrix=inverse_mass_matrix,
            data=data,
        )

    proposal_energy = hamiltonian_energy(proposal, inverse_mass_matrix)
    energy_error = proposal_energy - initial_energy
    finite_energy = jnp.isfinite(proposal_energy) & jnp.isfinite(energy_error)
    divergence_flag = ~finite_energy | (energy_error > threshold)
    acceptance_statistic = jnp.where(
        finite_energy,
        jnp.minimum(jnp.asarray(1.0, dtype=dtype), jnp.exp(-energy_error)),
        jnp.asarray(0.0, dtype=dtype),
    )
    accepted = (
        ~divergence_flag
        & (jr.uniform(accept_key, dtype=dtype) < acceptance_statistic)
    )
    next_state = SamplerState(
        position=jnp.where(accepted, proposal.position, state.position),
        potential_energy=jnp.where(
            accepted,
            proposal.potential_energy,
            state.potential_energy,
        ),
        potential_energy_grad=jnp.where(
            accepted,
            proposal.potential_energy_grad,
            state.potential_energy_grad,
        ),
    )
    info = TransitionInfo(
        acceptance_statistic=acceptance_statistic,
        divergence_flag=divergence_flag,
        realized_tree_depth=jnp.asarray(0, dtype=jnp.int32),
        leapfrog_count=jnp.asarray(num_leapfrog_steps, dtype=jnp.int32),
        energy_error=jnp.where(
            finite_energy,
            energy_error,
            jnp.asarray(jnp.inf, dtype=dtype),
        ),
        gradient_norm=jnp.linalg.norm(next_state.potential_energy_grad),
        max_tree_depth_hit=jnp.asarray(False),
    )
    return next_state, info, key


def fixed_hmc_transition(
    model: BenchmarkModel,
    state: SamplerState,
    rng_keys: Array,
    *,
    step_size: float | Array,
    num_leapfrog_steps: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Apply fixed-length HMC independently to each chain."""
    num_chains = state.position.shape[0]
    if rng_keys.shape != (num_chains, 2):
        raise ValueError(
            "rng_keys must have shape (num_chains, 2); "
            f"got {rng_keys.shape} for {num_chains} chains"
        )

    def transition_chain(
        chain_state: SamplerState,
        chain_key: Array,
    ) -> tuple[SamplerState, TransitionInfo, Array]:
        return one_chain_fixed_hmc_transition(
            model,
            chain_state,
            chain_key,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
        )

    return jax.vmap(transition_chain)(state, rng_keys)


def run_fixed_hmc(
    model: BenchmarkModel,
    initial_positions: Array,
    rng_key: Array,
    *,
    num_steps: int,
    step_size: float | Array,
    num_leapfrog_steps: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    dtype: Any = jnp.float32,
    data: Any | None = None,
) -> FixedHmcRunResult:
    """Run a tiny fixed-length HMC multi-chain loop."""
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
        state, info, rng_keys = fixed_hmc_transition(
            model,
            state,
            rng_keys,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=divergence_threshold,
            data=data,
        )
        trace_positions.append(state.position)
        infos.append(info)

    stacked_info = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *infos)
    return FixedHmcRunResult(
        initial_state=initial_state,
        final_state=state,
        initial_rng_keys=initial_rng_keys,
        final_rng_keys=rng_keys,
        trace_positions=jnp.stack(trace_positions),
        transition_info=stacked_info,
        num_leapfrog_steps=num_leapfrog_steps,
    )
