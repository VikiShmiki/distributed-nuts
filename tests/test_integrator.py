from __future__ import annotations

import math

import jax.numpy as jnp
import jax.random as jr

from abnuts.models.funnel import FunnelModel
from abnuts.nuts.hamiltonian import hamiltonian_energy, kinetic_energy, potential_energy_and_grad
from abnuts.nuts.integrator import leapfrog_step, new_integrator_state


def test_hamiltonian_quantities_are_finite_on_funnel() -> None:
    model = FunnelModel(dimension=4)
    position = jnp.asarray(model.initial_position(key=11, num_chains=1)[0], dtype=jnp.float32)
    momentum = jnp.linspace(-0.3, 0.3, model.dimension, dtype=jnp.float32)

    potential, grad = potential_energy_and_grad(model, position)
    state = new_integrator_state(model, position, momentum)
    total_energy = hamiltonian_energy(state)

    assert grad.shape == position.shape
    assert math.isfinite(float(potential))
    assert math.isfinite(float(kinetic_energy(momentum)))
    assert math.isfinite(float(total_energy))


def test_leapfrog_forward_reverse_reconstructs_initial_state() -> None:
    model = FunnelModel(dimension=5)
    position = jnp.asarray(model.initial_position(key=3, num_chains=1)[0], dtype=jnp.float32)
    momentum = jr.normal(jr.PRNGKey(19), shape=position.shape, dtype=position.dtype)

    initial = new_integrator_state(model, position, momentum)
    forward = leapfrog_step(model, initial, step_size=1e-4)
    reverse_start = forward.replace(momentum=-forward.momentum)
    reversed_state = leapfrog_step(model, reverse_start, step_size=1e-4)

    assert jnp.allclose(reversed_state.position, initial.position, rtol=1e-6, atol=1e-6)
    assert jnp.allclose(-reversed_state.momentum, initial.momentum, rtol=1e-6, atol=1e-6)
