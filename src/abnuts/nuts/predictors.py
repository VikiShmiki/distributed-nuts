"""Scheduling predictors for bucketed NUTS."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from abnuts.models.base import BenchmarkModel
from abnuts.nuts.hamiltonian import potential_energy

HVP_PREDICTOR_MODES = frozenset({"hvp", "hybrid"})
PREDICTOR_MODES = frozenset(
    {"none", "random", "history", "last_depth", *HVP_PREDICTOR_MODES}
)


class PredictorState(NamedTuple):
    """Persistent per-chain predictor statistics."""

    history_ema_work: Array
    hvp_work: Array
    last_realized_work: Array
    num_updates: Array


def new_predictor_state(
    num_chains: int,
    *,
    initial_work: float | Array = 1.0,
    dtype: Any = jnp.float32,
) -> PredictorState:
    """Create predictor state for ``num_chains`` independent chains."""
    if num_chains <= 0:
        raise ValueError(f"num_chains must be positive, got {num_chains!r}")

    initial = _as_chain_vector(
        initial_work,
        num_chains=num_chains,
        dtype=dtype,
        name="initial_work",
    )
    return PredictorState(
        history_ema_work=initial,
        hvp_work=initial,
        last_realized_work=jnp.zeros_like(initial),
        num_updates=jnp.asarray(0, dtype=jnp.int32),
    )


def predict_work(
    mode: str,
    state: PredictorState,
    *,
    rng_key: Array | None = None,
) -> Array:
    """Return one predicted work score per chain for a supported predictor mode."""
    _validate_mode(mode)

    if mode == "none":
        return jnp.ones_like(state.history_ema_work)
    if mode == "history":
        return state.history_ema_work
    if mode == "last_depth":
        return jnp.where(
            state.num_updates > 0,
            state.last_realized_work,
            jnp.ones_like(state.last_realized_work),
        )
    if mode == "hvp":
        return state.hvp_work
    if mode == "hybrid":
        return jnp.maximum(state.history_ema_work, state.hvp_work)

    if rng_key is None:
        raise ValueError("random predictor requires rng_key")
    return jr.uniform(
        rng_key,
        shape=state.history_ema_work.shape,
        dtype=state.history_ema_work.dtype,
    )


def predictor_uses_hvp(mode: str) -> bool:
    """Return true when ``mode`` needs a fresh HVP probe before planning."""
    _validate_mode(mode)
    return mode in HVP_PREDICTOR_MODES


def hvp_curvature_work(
    model: BenchmarkModel,
    positions: Array,
    *,
    data: Any | None = None,
    minimum_work: float = 1.0,
) -> Array:
    """Estimate per-chain work from a deterministic Hessian-vector product.

    The probe is intentionally deterministic so the predictor cannot consume or
    perturb per-chain transition RNG.  The returned scale is positive and lives
    on the same "larger means more work" axis as the history predictor.
    """
    if minimum_work < 0.0:
        raise ValueError(f"minimum_work must be nonnegative, got {minimum_work!r}")

    position_array = jnp.asarray(positions)
    if position_array.ndim != 2:
        raise ValueError(
            "positions must have shape (num_chains, dimension); "
            f"got {position_array.shape}"
        )
    if position_array.shape[0] <= 0:
        raise ValueError("positions must contain at least one chain")

    dtype = position_array.dtype
    probe = _deterministic_probe(position_array.shape[1], dtype=dtype)
    minimum = jnp.asarray(minimum_work, dtype=dtype)

    def potential_at(position: Array) -> Array:
        return potential_energy(model, position, data=data)

    potential_grad = jax.grad(potential_at)

    def chain_work(position: Array) -> Array:
        _, hvp = jax.jvp(potential_grad, (position,), (probe,))
        curvature = jnp.linalg.norm(hvp)
        return minimum + jnp.log1p(curvature)

    return jax.vmap(chain_work)(position_array)


def update_hvp_work(state: PredictorState, hvp_work: Array) -> PredictorState:
    """Store the latest HVP work estimate in predictor state."""
    hvp = _as_chain_vector(
        hvp_work,
        num_chains=state.history_ema_work.shape[0],
        dtype=state.history_ema_work.dtype,
        name="hvp_work",
    )
    return PredictorState(
        history_ema_work=state.history_ema_work,
        hvp_work=hvp,
        last_realized_work=state.last_realized_work,
        num_updates=state.num_updates,
    )


def update_predictor_state(
    state: PredictorState,
    realized_work: Array,
    *,
    beta: float = 0.9,
) -> PredictorState:
    """Update history statistics from realized per-chain depth or work."""
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta!r}")

    realized = _as_chain_vector(
        realized_work,
        num_chains=state.history_ema_work.shape[0],
        dtype=state.history_ema_work.dtype,
        name="realized_work",
    )
    beta_value = jnp.asarray(beta, dtype=state.history_ema_work.dtype)
    ema = beta_value * state.history_ema_work + (1.0 - beta_value) * realized
    return PredictorState(
        history_ema_work=ema,
        hvp_work=state.hvp_work,
        last_realized_work=realized,
        num_updates=state.num_updates + jnp.asarray(1, dtype=state.num_updates.dtype),
    )


def _validate_mode(mode: str) -> None:
    if mode not in PREDICTOR_MODES:
        valid_modes = ", ".join(sorted(PREDICTOR_MODES))
        raise ValueError(f"unknown predictor mode {mode!r}; expected one of {valid_modes}")


def _as_chain_vector(
    value: float | Array,
    *,
    num_chains: int,
    dtype: Any,
    name: str,
) -> Array:
    array = jnp.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return jnp.full((num_chains,), array, dtype=dtype)
    if array.shape != (num_chains,):
        raise ValueError(f"{name} must be scalar or have shape ({num_chains},); got {array.shape}")
    return array


def _deterministic_probe(dimension: int, *, dtype: Any) -> Array:
    if dimension <= 0:
        raise ValueError(f"dimension must be positive, got {dimension!r}")
    raw = jnp.arange(1, dimension + 1, dtype=dtype)
    return raw / jnp.linalg.norm(raw)
