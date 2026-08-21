"""One-chain iterative NUTS transition with diagnostic metrics.

Two implementations of the same transition live here:

``one_chain_nuts_transition``
    The default. Trajectory doubling and subtree construction use
    ``lax.while_loop``, so a chain stops executing leapfrog steps as soon as it
    diverges, U-turns, or reaches ``max_tree_depth``. Executed work is therefore
    a function of the *realized* tree depth.

``one_chain_nuts_transition_unrolled``
    The previous implementation, kept as a differential-testing reference. Both
    loops are Python loops unrolled at trace time, and ``active`` /
    ``should_step`` only mask results through ``jnp.where``. Every chain always
    executes the full ``2**max_tree_depth - 1`` leapfrog budget regardless of
    what it actually does.

The two are semantically identical: the leapfrog integrator, U-turn criterion,
divergence logic, max-depth logic, and per-chain RNG sequence are the same. The
skipped iterations of the unrolled version are exact no-ops (they add ``0`` to
counters and sums, take ``maximum`` against ``0.0``, and select the unchanged
branch), so removing them cannot change a result. See
``tests/test_transition_control_flow.py``.

Why the unrolled version had to go: under ``jax.vmap``, ``lax.while_loop`` runs
while *any* lane's predicate holds and freezes finished lanes, so a vectorized
group costs the group's maximum realized depth rather than ``max_tree_depth``.
That difference is the straggler waste the bucketing scheduler exists to
reclaim; with the unrolled version there was none, and no scheduler could ever
win. See the 2026-08-09 blocking finding in ``STATUS.md``.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import Array, lax

from abnuts.models.base import BenchmarkModel
from abnuts.nuts.hamiltonian import hamiltonian_energy, velocity_from_momentum
from abnuts.nuts.integrator import leapfrog_step, new_integrator_state
from abnuts.nuts.state import IntegratorState, SamplerState


class TransitionInfo(NamedTuple):
    """Diagnostics emitted by one NUTS transition."""

    acceptance_statistic: Array
    divergence_flag: Array
    realized_tree_depth: Array
    leapfrog_count: Array
    energy_error: Array
    gradient_norm: Array
    max_tree_depth_hit: Array


class _SubtreeResult(NamedTuple):
    endpoint: IntegratorState
    proposal: IntegratorState
    valid_count: Array
    alpha_sum: Array
    alpha_count: Array
    divergence_flag: Array
    turning: Array
    leapfrog_count: Array
    max_energy_error: Array


class _SubtreeCarry(NamedTuple):
    """Loop state for one ``lax.while_loop`` subtree build.

    ``checkpoint_position`` and ``checkpoint_momentum`` are the U-turn
    checkpoint stack, one entry per binary-tree level. See
    ``_build_subtree_control_flow``.
    """

    step_index: Array
    rng_key: Array
    checkpoint_position: Array
    checkpoint_momentum: Array
    current: IntegratorState
    proposal: IntegratorState
    valid_count: Array
    alpha_sum: Array
    alpha_count: Array
    divergence_flag: Array
    turning: Array
    leapfrog_count: Array
    max_energy_error: Array


class _TrajectoryCarry(NamedTuple):
    """Loop state for the ``lax.while_loop`` over trajectory doublings.

    ``depth`` doubles as the realized tree depth: the loop runs exactly while
    the chain is active, so on exit ``depth`` equals the number of attempted
    doublings, which is what the unrolled version accumulates into
    ``realized_tree_depth``.
    """

    depth: Array
    rng_key: Array
    left: IntegratorState
    right: IntegratorState
    proposal: IntegratorState
    valid_count: Array
    alpha_sum: Array
    alpha_count: Array
    divergence_flag: Array
    leapfrog_count: Array
    max_energy_error: Array
    active: Array


def new_sampler_state(
    model: BenchmarkModel,
    position: Array,
    data: Any | None = None,
) -> SamplerState:
    """Construct persistent sampler state by evaluating potential and gradient."""
    position_array = jnp.asarray(position)
    zero_momentum = jnp.zeros_like(position_array)
    integrator_state = new_integrator_state(model, position_array, zero_momentum, data=data)
    return SamplerState(
        position=integrator_state.position,
        potential_energy=integrator_state.potential_energy,
        potential_energy_grad=integrator_state.potential_energy_grad,
    )


def one_chain_nuts_transition(
    model: BenchmarkModel,
    state: SamplerState,
    rng_key: Array,
    *,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Run one fixed-step NUTS transition for a single chain.

    This transition samples a fresh momentum, builds a binary trajectory by
    iterative depth doublings, and chooses among slice-valid states with
    reservoir sampling. Adaptation is intentionally out of scope for this
    primitive.

    Both the doubling loop and the subtree loop are ``lax.while_loop``s with
    data-dependent exit conditions, so executed leapfrog work scales with the
    realized tree depth rather than with ``max_tree_depth``.
    """
    if max_tree_depth <= 0:
        raise ValueError(f"max_tree_depth must be positive, got {max_tree_depth!r}")

    dtype = state.position.dtype
    key, initial, initial_energy, log_slice, step, threshold = _begin_trajectory(
        state,
        rng_key,
        step_size=step_size,
        divergence_threshold=divergence_threshold,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    def doubling_cond(carry: _TrajectoryCarry) -> Array:
        return carry.active & (carry.depth < max_tree_depth)

    def doubling_body(carry: _TrajectoryCarry) -> _TrajectoryCarry:
        direction_key, proposal_key, subtree_key, next_key = jr.split(carry.rng_key, 4)
        direction_is_left = jr.bernoulli(direction_key)
        start = _select_integrator_state(direction_is_left, carry.left, carry.right)
        signed_step = jnp.where(direction_is_left, -step, step)

        # 2**depth, as int32. max_tree_depth is bounded well below 31 in
        # practice; the unrolled reference would be untraceable long before the
        # shift could overflow.
        num_steps = jnp.left_shift(jnp.asarray(1, dtype=jnp.int32), carry.depth)

        # Under vmap this body also runs for lanes that already exited, whose
        # results the while_loop batching rule discards. Passing the loop
        # predicate rather than a bare True keeps those lanes from doing real
        # leapfrog work; inside a genuine iteration the predicate is True, so
        # this does not change any result.
        subtree = _build_subtree_control_flow(
            model=model,
            start=start,
            rng_key=subtree_key,
            log_slice=log_slice,
            initial_energy=initial_energy,
            step_size=signed_step,
            num_steps=num_steps,
            active=doubling_cond(carry),
            integrate_backward=direction_is_left,
            max_levels=max_tree_depth,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=threshold,
            data=data,
        )

        # ``carry.active`` is True for every genuine iteration, so the unrolled
        # version's ``depth_was_attempted`` conjunctions all collapse away here.
        left = _select_integrator_state(direction_is_left, subtree.endpoint, carry.left)
        right = _select_integrator_state(~direction_is_left, subtree.endpoint, carry.right)

        subtree_is_usable = ~subtree.divergence_flag & ~subtree.turning
        usable_subtree_count = jnp.where(
            subtree_is_usable,
            subtree.valid_count,
            jnp.asarray(0, dtype=jnp.int32),
        )
        candidate_count = carry.valid_count + usable_subtree_count
        choose_probability = jnp.where(
            candidate_count > 0,
            usable_subtree_count.astype(dtype) / candidate_count.astype(dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        choose_subtree = (
            subtree_is_usable
            & (subtree.valid_count > 0)
            & (jr.uniform(proposal_key, dtype=dtype) < choose_probability)
        )
        proposal = _select_integrator_state(choose_subtree, subtree.proposal, carry.proposal)

        global_turning = _is_turning(left, right, inverse_mass_matrix)

        return _TrajectoryCarry(
            depth=carry.depth + 1,
            rng_key=next_key,
            left=left,
            right=right,
            proposal=proposal,
            valid_count=candidate_count,
            alpha_sum=carry.alpha_sum + subtree.alpha_sum,
            alpha_count=carry.alpha_count + subtree.alpha_count,
            divergence_flag=carry.divergence_flag | subtree.divergence_flag,
            leapfrog_count=carry.leapfrog_count + subtree.leapfrog_count,
            max_energy_error=jnp.maximum(carry.max_energy_error, subtree.max_energy_error),
            active=~subtree.divergence_flag & ~subtree.turning & ~global_turning,
        )

    initial_carry = _TrajectoryCarry(
        depth=jnp.asarray(0, dtype=jnp.int32),
        rng_key=key,
        left=initial,
        right=initial,
        proposal=initial,
        valid_count=jnp.asarray(1, dtype=jnp.int32),
        alpha_sum=jnp.asarray(0.0, dtype=dtype),
        alpha_count=jnp.asarray(0, dtype=jnp.int32),
        divergence_flag=jnp.asarray(False),
        leapfrog_count=jnp.asarray(0, dtype=jnp.int32),
        max_energy_error=jnp.asarray(0.0, dtype=dtype),
        active=jnp.asarray(True),
    )
    final = lax.while_loop(doubling_cond, doubling_body, initial_carry)

    # The unrolled version splits the trajectory key once per depth for all
    # max_tree_depth depths, whether or not the chain is still active, and
    # returns the last one. Exiting early must not change the chain's RNG
    # sequence, so burn the splits the early exit skipped.
    next_rng_key = _advance_trajectory_key(final.rng_key, final.depth, max_tree_depth)

    return _finish_trajectory(
        proposal=final.proposal,
        initial_energy=initial_energy,
        alpha_sum=final.alpha_sum,
        alpha_count=final.alpha_count,
        divergence_flag=final.divergence_flag,
        realized_tree_depth=final.depth,
        leapfrog_count=final.leapfrog_count,
        max_energy_error=final.max_energy_error,
        max_tree_depth_hit=final.active,
        inverse_mass_matrix=inverse_mass_matrix,
        dtype=dtype,
    ) + (next_rng_key,)


def one_chain_nuts_transition_unrolled(
    model: BenchmarkModel,
    state: SamplerState,
    rng_key: Array,
    *,
    step_size: float | Array,
    max_tree_depth: int,
    inverse_mass_matrix: Array | float | None = None,
    divergence_threshold: float = 1000.0,
    data: Any | None = None,
) -> tuple[SamplerState, TransitionInfo, Array]:
    """Trace-time-unrolled reference implementation of the same transition.

    Kept only as the differential-testing reference for
    ``one_chain_nuts_transition``. Do not use it in samplers or benchmarks: it
    executes the full ``2**max_tree_depth - 1`` leapfrog budget for every chain
    regardless of realized depth, which both erases the work heterogeneity the
    scheduler targets and makes compile time grow exponentially in
    ``max_tree_depth``.
    """
    if max_tree_depth <= 0:
        raise ValueError(f"max_tree_depth must be positive, got {max_tree_depth!r}")

    dtype = state.position.dtype
    key, initial, initial_energy, log_slice, step, threshold = _begin_trajectory(
        state,
        rng_key,
        step_size=step_size,
        divergence_threshold=divergence_threshold,
        inverse_mass_matrix=inverse_mass_matrix,
    )

    left = initial
    right = initial
    proposal = initial
    valid_count = jnp.asarray(1, dtype=jnp.int32)
    alpha_sum = jnp.asarray(0.0, dtype=dtype)
    alpha_count = jnp.asarray(0, dtype=jnp.int32)
    divergence_flag = jnp.asarray(False)
    leapfrog_count = jnp.asarray(0, dtype=jnp.int32)
    max_energy_error = jnp.asarray(0.0, dtype=dtype)
    realized_tree_depth = jnp.asarray(0, dtype=jnp.int32)
    active = jnp.asarray(True)

    for depth in range(max_tree_depth):
        direction_key, proposal_key, subtree_key, key = jr.split(key, 4)
        direction_is_left = jr.bernoulli(direction_key)
        start = _select_integrator_state(direction_is_left, left, right)
        signed_step = jnp.where(direction_is_left, -step, step)

        subtree = _build_subtree_unrolled(
            model=model,
            start=start,
            rng_key=subtree_key,
            log_slice=log_slice,
            initial_energy=initial_energy,
            step_size=signed_step,
            num_steps=1 << depth,
            active=active,
            integrate_backward=direction_is_left,
            inverse_mass_matrix=inverse_mass_matrix,
            divergence_threshold=threshold,
            data=data,
        )

        depth_was_attempted = active
        realized_tree_depth += depth_was_attempted.astype(jnp.int32)

        update_left = depth_was_attempted & direction_is_left
        update_right = depth_was_attempted & ~direction_is_left
        left = _select_integrator_state(update_left, subtree.endpoint, left)
        right = _select_integrator_state(update_right, subtree.endpoint, right)

        subtree_is_usable = (
            depth_was_attempted & ~subtree.divergence_flag & ~subtree.turning
        )
        usable_subtree_count = jnp.where(
            subtree_is_usable,
            subtree.valid_count,
            jnp.asarray(0, dtype=jnp.int32),
        )
        candidate_count = valid_count + usable_subtree_count
        choose_probability = jnp.where(
            candidate_count > 0,
            usable_subtree_count.astype(dtype) / candidate_count.astype(dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        choose_subtree = (
            subtree_is_usable
            & (subtree.valid_count > 0)
            & (jr.uniform(proposal_key, dtype=dtype) < choose_probability)
        )
        proposal = _select_integrator_state(choose_subtree, subtree.proposal, proposal)
        valid_count = candidate_count

        global_turning = _is_turning(left, right, inverse_mass_matrix)
        active = active & ~subtree.divergence_flag & ~subtree.turning & ~global_turning

        alpha_sum += subtree.alpha_sum
        alpha_count += subtree.alpha_count
        divergence_flag = divergence_flag | subtree.divergence_flag
        leapfrog_count += subtree.leapfrog_count
        max_energy_error = jnp.maximum(max_energy_error, subtree.max_energy_error)

    return _finish_trajectory(
        proposal=proposal,
        initial_energy=initial_energy,
        alpha_sum=alpha_sum,
        alpha_count=alpha_count,
        divergence_flag=divergence_flag,
        realized_tree_depth=realized_tree_depth,
        leapfrog_count=leapfrog_count,
        max_energy_error=max_energy_error,
        max_tree_depth_hit=active,
        inverse_mass_matrix=inverse_mass_matrix,
        dtype=dtype,
    ) + (key,)


def _begin_trajectory(
    state: SamplerState,
    rng_key: Array,
    *,
    step_size: float | Array,
    divergence_threshold: float,
    inverse_mass_matrix: Array | float | None,
) -> tuple[Array, IntegratorState, Array, Array, Array, Array]:
    """Draw momentum and the slice variable; shared by both implementations."""
    key, momentum_key, slice_key = jr.split(rng_key, 3)
    dtype = state.position.dtype
    step = jnp.asarray(step_size, dtype=dtype)
    threshold = jnp.asarray(divergence_threshold, dtype=dtype)

    momentum = jr.normal(momentum_key, shape=state.position.shape, dtype=dtype)
    initial = _integrator_from_sampler_state(state, momentum)
    initial_energy = hamiltonian_energy(initial, inverse_mass_matrix)
    log_slice = -initial_energy + _safe_log_uniform(slice_key, dtype)
    return key, initial, initial_energy, log_slice, step, threshold


def _finish_trajectory(
    *,
    proposal: IntegratorState,
    initial_energy: Array,
    alpha_sum: Array,
    alpha_count: Array,
    divergence_flag: Array,
    realized_tree_depth: Array,
    leapfrog_count: Array,
    max_energy_error: Array,
    max_tree_depth_hit: Array,
    inverse_mass_matrix: Array | float | None,
    dtype: Any,
) -> tuple[SamplerState, TransitionInfo]:
    """Build the sampler state and diagnostics; shared by both implementations."""
    next_state = SamplerState(
        position=proposal.position,
        potential_energy=proposal.potential_energy,
        potential_energy_grad=proposal.potential_energy_grad,
    )
    proposal_energy = hamiltonian_energy(proposal, inverse_mass_matrix)
    selected_energy_error = proposal_energy - initial_energy
    acceptance_statistic = jnp.where(
        alpha_count > 0,
        alpha_sum / alpha_count.astype(dtype),
        jnp.asarray(0.0, dtype=dtype),
    )
    info = TransitionInfo(
        acceptance_statistic=acceptance_statistic,
        divergence_flag=divergence_flag,
        realized_tree_depth=realized_tree_depth,
        leapfrog_count=leapfrog_count,
        energy_error=jnp.where(
            divergence_flag,
            jnp.maximum(jnp.abs(selected_energy_error), max_energy_error),
            selected_energy_error,
        ),
        gradient_norm=jnp.linalg.norm(next_state.potential_energy_grad),
        max_tree_depth_hit=max_tree_depth_hit,
    )
    return next_state, info


def _advance_trajectory_key(
    rng_key: Array,
    depth: Array,
    max_tree_depth: int,
) -> Array:
    """Burn the per-depth key splits an early exit skipped.

    The unrolled reference advances the trajectory key once per depth for all
    ``max_tree_depth`` depths regardless of whether the chain is still active,
    and returns the final key as the chain's next key. Early exit must not
    change that sequence, so this replays the remaining splits. Only PRNG work
    happens here, no leapfrog steps.
    """

    def cond(carry: tuple[Array, Array]) -> Array:
        remaining_depth, _ = carry
        return remaining_depth < max_tree_depth

    def body(carry: tuple[Array, Array]) -> tuple[Array, Array]:
        remaining_depth, key = carry
        return remaining_depth + 1, jr.split(key, 4)[3]

    return lax.while_loop(cond, body, (depth, rng_key))[1]


def _build_subtree_control_flow(
    *,
    model: BenchmarkModel,
    start: IntegratorState,
    rng_key: Array,
    log_slice: Array,
    initial_energy: Array,
    step_size: Array,
    num_steps: Array,
    active: Array,
    integrate_backward: Array,
    max_levels: int,
    inverse_mass_matrix: Array | float | None,
    divergence_threshold: Array,
    data: Any | None,
) -> _SubtreeResult:
    """Build one subtree, stopping at divergence, U-turn, or ``num_steps``.

    The loop predicate is exactly the unrolled version's ``should_step``, so
    every iteration this loop skips is one the unrolled version executed only to
    discard: it would have added ``0`` to every counter and sum, taken
    ``maximum`` against ``0.0``, and selected the unchanged branch of every
    state update.

    ``integrate_backward`` says which end of the span ``current`` is. The U-turn
    criterion is defined on ``(rightmost - leftmost) . momentum``, so the two
    endpoints must be passed in trajectory order, not in the order they were
    visited. See ``_oriented_span``.

    **Termination criterion.** Standard NUTS builds a subtree of ``2**j`` steps
    by recursively combining two subtrees of ``2**(j-1)`` and checking the
    U-turn across each combined span. The checks therefore land on *aligned*
    power-of-two blocks of steps, at every level of the binary tree.

    This loop reproduces that with a checkpoint stack, one entry per level:
    before step ``i`` a level-``k`` block opens whenever ``i % 2**k == 0``, and
    after step ``i`` it closes whenever ``(i + 1) % 2**k == 0``, at which point
    the span from its checkpoint to the current state is tested. Level 0 is a
    single leapfrog step and is not tested, matching the recursion's base case.

    The loop over levels is a Python loop of length ``max_levels`` because
    ``max_tree_depth`` is static; it costs a handful of selects per step rather
    than unrolling the trajectory.

    Not implemented, and deliberate: Betancourt's generalized criterion adds
    checks spanning the boundary between adjacent merged subtrees. This is the
    classic Hoffman-Gelman recursion.
    """
    dtype = start.position.dtype
    # Block size per binary-tree level, and a mask excluding level 0, whose
    # blocks are single leapfrog steps that the recursion's base case never
    # tests.
    block_sizes = jnp.left_shift(
        jnp.asarray(1, dtype=jnp.int32), jnp.arange(max_levels, dtype=jnp.int32)
    )
    testable_levels = jnp.arange(max_levels) >= 1

    def subtree_cond(carry: _SubtreeCarry) -> Array:
        return (
            active
            & (carry.step_index < num_steps)
            & ~carry.divergence_flag
            & ~carry.turning
        )

    def subtree_body(carry: _SubtreeCarry) -> _SubtreeCarry:
        step_key, next_rng_key = jr.split(carry.rng_key)

        # A level-k block opens at this step when i % 2**k == 0; its left
        # endpoint is the state before the step is taken.
        #
        # Vectorised over levels on purpose. Writing the levels one at a time
        # costs one scatter per level per leapfrog step, which under vmap is
        # levels x chains x dimension of memory traffic against a leapfrog step's
        # chains x dimension -- it dominated the step itself and cost ~25x on GPU.
        opens = (carry.step_index % block_sizes) == 0
        checkpoint_position = jnp.where(
            opens[:, None], carry.current.position[None, :], carry.checkpoint_position
        )
        checkpoint_momentum = jnp.where(
            opens[:, None], carry.current.momentum[None, :], carry.checkpoint_momentum
        )

        candidate = leapfrog_step(
            model,
            carry.current,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            data=data,
        )
        candidate_energy = hamiltonian_energy(candidate, inverse_mass_matrix)
        energy_error = candidate_energy - initial_energy
        finite_energy = jnp.isfinite(candidate_energy) & jnp.isfinite(energy_error)
        candidate_diverged = ~finite_energy | (energy_error > divergence_threshold)
        candidate_valid = finite_energy & ~candidate_diverged & (log_slice <= -candidate_energy)

        # The loop predicate is the unrolled version's ``should_step``, so it is
        # True throughout this body and its conjunctions collapse away.
        current = candidate

        alpha = jnp.where(
            finite_energy,
            jnp.minimum(jnp.asarray(1.0, dtype=dtype), jnp.exp(-energy_error)),
            jnp.asarray(0.0, dtype=dtype),
        )

        new_valid_count = carry.valid_count + candidate_valid.astype(jnp.int32)
        choose_probability = jnp.where(
            new_valid_count > 0,
            jnp.asarray(1.0, dtype=dtype) / new_valid_count.astype(dtype),
            jnp.asarray(0.0, dtype=dtype),
        )
        choose_candidate = candidate_valid & (
            jr.uniform(step_key, dtype=dtype) < choose_probability
        )
        proposal = _select_integrator_state(choose_candidate, candidate, carry.proposal)

        absolute_energy_error = jnp.where(
            finite_energy,
            jnp.abs(energy_error),
            jnp.asarray(jnp.inf, dtype=dtype),
        )

        # A level-k block closes at this step when (i + 1) % 2**k == 0. Test the
        # span from that block's checkpoint to the state just reached. Level 0
        # is one leapfrog step, which the recursion's base case never tests.
        #
        # All levels are tested in one batched dot product for the same reason
        # the opens are batched above.
        closes = ((carry.step_index + 1) % block_sizes) == 0
        within_subtree = block_sizes <= num_steps
        anchor_velocity = velocity_from_momentum(checkpoint_momentum, inverse_mass_matrix)
        current_velocity = velocity_from_momentum(current.momentum, inverse_mass_matrix)
        # Orientation: for a backward subtree the checkpoint is the right end.
        delta = jnp.where(
            integrate_backward,
            checkpoint_position - current.position[None, :],
            current.position[None, :] - checkpoint_position,
        )
        turns = (jnp.sum(delta * anchor_velocity, axis=-1) < 0.0) | (
            jnp.sum(delta * current_velocity[None, :], axis=-1) < 0.0
        )
        turning = carry.turning | (
            ~candidate_diverged
            & jnp.any(closes & within_subtree & testable_levels & turns)
        )

        return _SubtreeCarry(
            step_index=carry.step_index + 1,
            rng_key=next_rng_key,
            checkpoint_position=checkpoint_position,
            checkpoint_momentum=checkpoint_momentum,
            current=current,
            proposal=proposal,
            valid_count=new_valid_count,
            alpha_sum=carry.alpha_sum + alpha,
            alpha_count=carry.alpha_count + 1,
            divergence_flag=carry.divergence_flag | candidate_diverged,
            turning=turning,
            leapfrog_count=carry.leapfrog_count + 1,
            max_energy_error=jnp.maximum(carry.max_energy_error, absolute_energy_error),
        )

    initial_carry = _SubtreeCarry(
        step_index=jnp.asarray(0, dtype=jnp.int32),
        rng_key=rng_key,
        checkpoint_position=jnp.zeros((max_levels, *start.position.shape), dtype=dtype),
        checkpoint_momentum=jnp.zeros((max_levels, *start.momentum.shape), dtype=dtype),
        current=start,
        proposal=start,
        valid_count=jnp.asarray(0, dtype=jnp.int32),
        alpha_sum=jnp.asarray(0.0, dtype=dtype),
        alpha_count=jnp.asarray(0, dtype=jnp.int32),
        divergence_flag=jnp.asarray(False),
        turning=jnp.asarray(False),
        leapfrog_count=jnp.asarray(0, dtype=jnp.int32),
        max_energy_error=jnp.asarray(0.0, dtype=dtype),
    )
    final = lax.while_loop(subtree_cond, subtree_body, initial_carry)

    return _SubtreeResult(
        endpoint=final.current,
        proposal=final.proposal,
        valid_count=final.valid_count,
        alpha_sum=final.alpha_sum,
        alpha_count=final.alpha_count,
        divergence_flag=final.divergence_flag,
        turning=final.turning,
        leapfrog_count=final.leapfrog_count,
        max_energy_error=final.max_energy_error,
    )


def _integrator_from_sampler_state(
    state: SamplerState,
    momentum: Array,
) -> IntegratorState:
    return IntegratorState(
        position=state.position,
        momentum=momentum,
        potential_energy=state.potential_energy,
        potential_energy_grad=state.potential_energy_grad,
    )


def _build_subtree_unrolled(
    *,
    model: BenchmarkModel,
    start: IntegratorState,
    rng_key: Array,
    log_slice: Array,
    initial_energy: Array,
    step_size: Array,
    num_steps: int,
    active: Array,
    integrate_backward: Array,
    inverse_mass_matrix: Array | float | None,
    divergence_threshold: Array,
    data: Any | None,
) -> _SubtreeResult:
    """Trace-time-unrolled subtree build; reference for the control-flow version.

    ``num_steps`` and the step index are Python integers here, so the aligned
    power-of-two block structure is resolved at trace time with ordinary integer
    arithmetic instead of a traced checkpoint stack. That makes this an
    independent implementation of the same criterion rather than a restatement
    of the other one, which is what gives the differential test its value.
    """
    current = start
    proposal = start
    valid_count = jnp.asarray(0, dtype=jnp.int32)
    alpha_sum = jnp.asarray(0.0, dtype=start.position.dtype)
    alpha_count = jnp.asarray(0, dtype=jnp.int32)
    divergence_flag = jnp.asarray(False)
    turning = jnp.asarray(False)
    leapfrog_count = jnp.asarray(0, dtype=jnp.int32)
    max_energy_error = jnp.asarray(0.0, dtype=start.position.dtype)
    checkpoints: dict[int, tuple[Array, Array]] = {}
    num_levels = max(1, int(num_steps).bit_length())

    for step_index in range(num_steps):
        step_key, rng_key = jr.split(rng_key)
        for level in range(num_levels):
            if step_index % (1 << level) == 0:
                checkpoints[level] = (current.position, current.momentum)
        should_step = active & ~divergence_flag & ~turning
        candidate = leapfrog_step(
            model,
            current,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            data=data,
        )
        candidate_energy = hamiltonian_energy(candidate, inverse_mass_matrix)
        energy_error = candidate_energy - initial_energy
        finite_energy = jnp.isfinite(candidate_energy) & jnp.isfinite(energy_error)
        candidate_diverged = ~finite_energy | (energy_error > divergence_threshold)
        candidate_valid = finite_energy & ~candidate_diverged & (log_slice <= -candidate_energy)

        current = _select_integrator_state(should_step, candidate, current)
        leapfrog_count += should_step.astype(jnp.int32)

        alpha = jnp.where(
            finite_energy,
            jnp.minimum(jnp.asarray(1.0, dtype=start.position.dtype), jnp.exp(-energy_error)),
            jnp.asarray(0.0, dtype=start.position.dtype),
        )
        alpha_sum += jnp.where(
            should_step,
            alpha,
            jnp.asarray(0.0, dtype=start.position.dtype),
        )
        alpha_count += should_step.astype(jnp.int32)

        valid_step = should_step & candidate_valid
        new_valid_count = valid_count + valid_step.astype(jnp.int32)
        choose_probability = jnp.where(
            new_valid_count > 0,
            jnp.asarray(1.0, dtype=start.position.dtype)
            / new_valid_count.astype(start.position.dtype),
            jnp.asarray(0.0, dtype=start.position.dtype),
        )
        choose_candidate = valid_step & (
            jr.uniform(step_key, dtype=start.position.dtype) < choose_probability
        )
        proposal = _select_integrator_state(choose_candidate, candidate, proposal)
        valid_count = new_valid_count

        absolute_energy_error = jnp.where(
            finite_energy,
            jnp.abs(energy_error),
            jnp.asarray(jnp.inf, dtype=start.position.dtype),
        )
        max_energy_error = jnp.maximum(
            max_energy_error,
            jnp.where(
                should_step,
                absolute_energy_error,
                jnp.asarray(0.0, dtype=start.position.dtype),
            ),
        )
        divergence_flag = divergence_flag | (should_step & candidate_diverged)
        for level in range(1, num_levels):
            block = 1 << level
            if (step_index + 1) % block != 0 or block > num_steps:
                continue
            anchor_position, anchor_momentum = checkpoints[level]
            left_p, left_m, right_p, right_m = _oriented_span(
                integrate_backward,
                anchor_position,
                anchor_momentum,
                current.position,
                current.momentum,
            )
            turning = turning | (
                should_step
                & ~candidate_diverged
                & _is_turning_between(left_p, left_m, right_p, right_m, inverse_mass_matrix)
            )

    return _SubtreeResult(
        endpoint=current,
        proposal=proposal,
        valid_count=valid_count,
        alpha_sum=alpha_sum,
        alpha_count=alpha_count,
        divergence_flag=divergence_flag,
        turning=turning,
        leapfrog_count=leapfrog_count,
        max_energy_error=max_energy_error,
    )


def _safe_log_uniform(rng_key: Array, dtype: Any) -> Array:
    uniform = jr.uniform(rng_key, dtype=dtype)
    tiny = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    return jnp.log(jnp.maximum(uniform, tiny))


def _is_turning_between(
    left_position: Array,
    left_momentum: Array,
    right_position: Array,
    right_momentum: Array,
    inverse_mass_matrix: Array | float | None,
) -> Array:
    """U-turn test for a span given only its endpoints' position and momentum.

    Checkpoints only need these two fields, so storing whole integrator states
    for every tree level would waste memory proportional to chains x levels x
    dimension.
    """
    delta = right_position - left_position
    left_velocity = velocity_from_momentum(left_momentum, inverse_mass_matrix)
    right_velocity = velocity_from_momentum(right_momentum, inverse_mass_matrix)
    return (jnp.vdot(delta, left_velocity) < 0.0) | (
        jnp.vdot(delta, right_velocity) < 0.0
    )


def _oriented_span(
    integrate_backward: Array,
    anchor_position: Array,
    anchor_momentum: Array,
    current_position: Array,
    current_momentum: Array,
) -> tuple[Array, Array, Array, Array]:
    """Order a span's endpoints by trajectory position, not visit order."""
    left_position = jnp.where(integrate_backward, current_position, anchor_position)
    left_momentum = jnp.where(integrate_backward, current_momentum, anchor_momentum)
    right_position = jnp.where(integrate_backward, anchor_position, current_position)
    right_momentum = jnp.where(integrate_backward, anchor_momentum, current_momentum)
    return left_position, left_momentum, right_position, right_momentum


def _span_endpoints(
    integrate_backward: Array,
    start: IntegratorState,
    current: IntegratorState,
) -> tuple[IntegratorState, IntegratorState]:
    """Order a subtree's two endpoints by trajectory position, not visit order.

    A subtree is built by stepping away from ``start`` with a signed step size.
    Integrating forward puts ``current`` to the right of ``start``; integrating
    backward puts it to the left. The U-turn criterion is not symmetric in its
    arguments, so passing them in visit order makes every backward subtree
    report a U-turn on its first step: ``delta`` then points against the
    direction of travel and both dot products are negative regardless of the
    target's geometry.

    ``integrate_backward`` is a traced boolean because the direction is drawn at
    runtime, so this selects rather than branches.
    """
    left = _select_integrator_state(integrate_backward, current, start)
    right = _select_integrator_state(integrate_backward, start, current)
    return left, right


def _is_turning(
    left: IntegratorState,
    right: IntegratorState,
    inverse_mass_matrix: Array | float | None,
) -> Array:
    delta = right.position - left.position
    left_velocity = velocity_from_momentum(left.momentum, inverse_mass_matrix)
    right_velocity = velocity_from_momentum(right.momentum, inverse_mass_matrix)
    return (jnp.vdot(delta, left_velocity) < 0.0) | (
        jnp.vdot(delta, right_velocity) < 0.0
    )


def _select_integrator_state(
    predicate: Array,
    if_true: IntegratorState,
    if_false: IntegratorState,
) -> IntegratorState:
    return IntegratorState(
        position=jnp.where(predicate, if_true.position, if_false.position),
        momentum=jnp.where(predicate, if_true.momentum, if_false.momentum),
        potential_energy=jnp.where(
            predicate,
            if_true.potential_energy,
            if_false.potential_energy,
        ),
        potential_energy_grad=jnp.where(
            predicate,
            if_true.potential_energy_grad,
            if_false.potential_energy_grad,
        ),
    )
