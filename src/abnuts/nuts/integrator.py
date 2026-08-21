"""Leapfrog integration for Hamiltonian dynamics."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from jax import Array

from abnuts.models.base import BenchmarkModel
from abnuts.nuts.hamiltonian import potential_energy_and_grad, velocity_from_momentum
from abnuts.nuts.state import IntegratorState


def new_integrator_state(
    model: BenchmarkModel,
    position: Array,
    momentum: Array,
    data: Any | None = None,
) -> IntegratorState:
    """Construct an integrator state by evaluating potential energy and gradient."""
    position_array = jnp.asarray(position)
    momentum_array = jnp.asarray(momentum, dtype=position_array.dtype)
    if position_array.shape != momentum_array.shape:
        raise ValueError(
            "position and momentum must have matching shapes; "
            f"got {position_array.shape} and {momentum_array.shape}"
        )

    value, grad = potential_energy_and_grad(model, position_array, data=data)
    return IntegratorState(
        position=position_array,
        momentum=momentum_array,
        potential_energy=value,
        potential_energy_grad=grad,
    )


def leapfrog_step(
    model: BenchmarkModel,
    state: IntegratorState,
    step_size: float | Array,
    inverse_mass_matrix: Array | float | None = None,
    data: Any | None = None,
) -> IntegratorState:
    """Take one reversible leapfrog step for a single chain."""
    step = jnp.asarray(step_size, dtype=state.position.dtype)
    half_momentum = state.momentum - 0.5 * step * state.potential_energy_grad
    next_position = state.position + step * velocity_from_momentum(
        half_momentum,
        inverse_mass_matrix,
    )
    next_potential, next_grad = potential_energy_and_grad(model, next_position, data=data)
    next_momentum = half_momentum - 0.5 * step * next_grad
    return IntegratorState(
        position=next_position,
        momentum=next_momentum,
        potential_energy=next_potential,
        potential_energy_grad=next_grad,
    )
