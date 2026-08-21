"""State containers shared by Hamiltonian integrators."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array


class IntegratorState(NamedTuple):
    """Position, momentum, potential energy, and gradient for one chain."""

    position: Array
    momentum: Array
    potential_energy: Array
    potential_energy_grad: Array

    def replace(
        self,
        *,
        position: Array | None = None,
        momentum: Array | None = None,
        potential_energy: Array | None = None,
        potential_energy_grad: Array | None = None,
    ) -> IntegratorState:
        """Return a copy with selected fields replaced."""
        new_potential_energy = (
            self.potential_energy if potential_energy is None else potential_energy
        )
        return IntegratorState(
            position=self.position if position is None else position,
            momentum=self.momentum if momentum is None else momentum,
            potential_energy=new_potential_energy,
            potential_energy_grad=(
                self.potential_energy_grad
                if potential_energy_grad is None
                else potential_energy_grad
            ),
        )


class SamplerState(NamedTuple):
    """Persistent one-chain state carried between NUTS transitions."""

    position: Array
    potential_energy: Array
    potential_energy_grad: Array

    def replace(
        self,
        *,
        position: Array | None = None,
        potential_energy: Array | None = None,
        potential_energy_grad: Array | None = None,
    ) -> SamplerState:
        """Return a copy with selected fields replaced."""
        return SamplerState(
            position=self.position if position is None else position,
            potential_energy=(
                self.potential_energy if potential_energy is None else potential_energy
            ),
            potential_energy_grad=(
                self.potential_energy_grad
                if potential_energy_grad is None
                else potential_energy_grad
            ),
        )
