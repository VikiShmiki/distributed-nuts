"""Hamiltonian energy primitives for diagonal-mass HMC/NUTS."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from abnuts.models.base import BenchmarkModel
from abnuts.nuts.state import IntegratorState


def potential_energy(
    model: BenchmarkModel,
    position: Array,
    data: Any | None = None,
) -> Array:
    """Return the potential energy ``-log_prob(position)`` for one chain."""
    return -jnp.asarray(model.log_prob(position, data=data))


def potential_energy_and_grad(
    model: BenchmarkModel,
    position: Array,
    data: Any | None = None,
) -> tuple[Array, Array]:
    """Return potential energy and its gradient with respect to position."""

    def potential_at(q: Array) -> Array:
        return potential_energy(model, q, data=data)

    value, grad = jax.value_and_grad(potential_at)(jnp.asarray(position))
    return value, grad


def velocity_from_momentum(
    momentum: Array,
    inverse_mass_matrix: Array | float | None = None,
) -> Array:
    """Convert momentum to velocity under a diagonal inverse mass matrix."""
    momentum_array = jnp.asarray(momentum)
    if inverse_mass_matrix is None:
        return momentum_array

    inverse_mass = jnp.asarray(inverse_mass_matrix, dtype=momentum_array.dtype)
    if inverse_mass.ndim > 0 and inverse_mass.shape != momentum_array.shape:
        raise ValueError(
            "inverse_mass_matrix must be scalar or have the same shape as momentum; "
            f"got {inverse_mass.shape} and {momentum_array.shape}"
        )
    return inverse_mass * momentum_array


def kinetic_energy(
    momentum: Array,
    inverse_mass_matrix: Array | float | None = None,
) -> Array:
    """Return ``0.5 * p.T @ inv_mass @ p`` for a diagonal mass matrix."""
    momentum_array = jnp.asarray(momentum)
    velocity = velocity_from_momentum(momentum_array, inverse_mass_matrix)
    return 0.5 * jnp.sum(momentum_array * velocity)


def hamiltonian_energy(
    state: IntegratorState,
    inverse_mass_matrix: Array | float | None = None,
) -> Array:
    """Return the total Hamiltonian energy for an integrator state."""
    return state.potential_energy + kinetic_energy(state.momentum, inverse_mass_matrix)
