"""T39 gate: the NUTS transition must have data-dependent control flow.

Two things are checked here, and both matter to the research claim.

Semantics
    ``one_chain_nuts_transition`` (``lax.while_loop``) must produce the same
    transition as ``one_chain_nuts_transition_unrolled`` (the previous
    trace-time-unrolled implementation). Discrete metrics and RNG keys must be
    exact; floats use the same strict ``1e-5`` tolerance T33/T35 already
    document for compiled-versus-eager float32 reordering.

Mechanism
    Executed work must depend on the *realized* tree depth. Before T39 it did
    not: a batch whose chains all stopped at depth 1 cost the same wall time as
    a batch reaching depth 6, because every chain always executed the full
    ``2**max_tree_depth - 1`` leapfrog budget. That left no straggler waste for
    the bucketing scheduler to reclaim, which is why every bucketed
    configuration was slower than monolithic. See the 2026-08-09 blocking
    finding in ``STATUS.md``. These tests exist so that defect cannot return
    silently.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from abnuts.blocking import block_until_ready_tree
from abnuts.models.funnel import FunnelModel
from abnuts.nuts.monolithic import jit_monolithic_transition, new_multi_chain_state
from abnuts.nuts.transition import (
    new_sampler_state,
    one_chain_nuts_transition,
    one_chain_nuts_transition_unrolled,
)

# Same strict tolerance T33 and T35 use for compiled-versus-eager float32.
FLOAT_ATOL = 1e-5
FLOAT_RTOL = 1e-5

EXACT_METRICS = (
    "realized_tree_depth",
    "leapfrog_count",
    "divergence_flag",
    "max_tree_depth_hit",
)
FLOAT_METRICS = ("acceptance_statistic", "energy_error", "gradient_norm")

# step_size drives the regime: tiny steps build deep trees, huge steps diverge
# immediately at depth 1.
DEEP_STEP_SIZE = 0.001
SHALLOW_STEP_SIZE = 25.0
MIXED_STEP_SIZE = 0.05


def _count_primitive(jaxpr: Any, name: str) -> int:
    """Count occurrences of a primitive, descending into nested jaxprs."""
    total = 0
    for equation in jaxpr.eqns:
        if equation.primitive.name == name:
            total += 1
        for parameter in equation.params.values():
            closed = getattr(parameter, "jaxpr", parameter)
            if hasattr(closed, "eqns"):
                total += _count_primitive(closed, name)
            elif isinstance(parameter, tuple | list):
                for item in parameter:
                    inner = getattr(item, "jaxpr", item)
                    if hasattr(inner, "eqns"):
                        total += _count_primitive(inner, name)
    return total


def _transition_jaxpr(transition: Any, max_tree_depth: int = 4) -> Any:
    model = FunnelModel(dimension=4)
    position = jnp.asarray(model.initial_position(key=5, num_chains=1)[0], dtype=jnp.float32)
    state = new_sampler_state(model, position)

    def run(sampler_state: Any, rng_key: Any) -> Any:
        return transition(
            model,
            sampler_state,
            rng_key,
            step_size=MIXED_STEP_SIZE,
            max_tree_depth=max_tree_depth,
        )

    return jax.make_jaxpr(run)(state, jr.PRNGKey(0)).jaxpr


def test_transition_uses_data_dependent_control_flow() -> None:
    """The default transition must contain while loops; the reference must not.

    This is the structural half of the T39 gate. It is deterministic, so it
    fails loudly if someone reintroduces an unrolled hot path, regardless of
    machine timing noise.
    """
    control_flow_whiles = _count_primitive(_transition_jaxpr(one_chain_nuts_transition), "while")
    unrolled_whiles = _count_primitive(
        _transition_jaxpr(one_chain_nuts_transition_unrolled),
        "while",
    )

    # Doubling loop, subtree loop, and the key-advance loop.
    assert control_flow_whiles >= 2, (
        "one_chain_nuts_transition must use lax.while_loop for trajectory "
        f"doubling and subtree construction; found {control_flow_whiles} while primitives"
    )
    assert unrolled_whiles == 0


@pytest.mark.parametrize("step_size", [MIXED_STEP_SIZE, DEEP_STEP_SIZE, SHALLOW_STEP_SIZE])
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_control_flow_transition_matches_unrolled_transition(
    step_size: float,
    seed: int,
) -> None:
    """Differential gate: early exit must not change the transition.

    Every iteration the control-flow version skips is one the unrolled version
    executed only to discard, so positions, metrics, and the chain's RNG
    sequence must agree.
    """
    model = FunnelModel(dimension=5)
    position = jr.normal(jr.PRNGKey(seed + 100), (5,), dtype=jnp.float32)
    state = new_sampler_state(model, position)
    key = jr.PRNGKey(seed)

    control_state, control_info, control_key = one_chain_nuts_transition(
        model,
        state,
        key,
        step_size=step_size,
        max_tree_depth=6,
    )
    unrolled_state, unrolled_info, unrolled_key = one_chain_nuts_transition_unrolled(
        model,
        state,
        key,
        step_size=step_size,
        max_tree_depth=6,
    )

    # The chain's RNG sequence must be preserved exactly. The control-flow
    # version exits early but still burns the per-depth key splits the unrolled
    # version consumed, so the next key is identical.
    assert jnp.array_equal(control_key, unrolled_key)

    for metric in EXACT_METRICS:
        assert jnp.array_equal(getattr(control_info, metric), getattr(unrolled_info, metric)), (
            f"{metric} must match exactly"
        )

    assert jnp.allclose(
        control_state.position,
        unrolled_state.position,
        atol=FLOAT_ATOL,
        rtol=FLOAT_RTOL,
    )
    for metric in FLOAT_METRICS:
        assert jnp.allclose(
            getattr(control_info, metric),
            getattr(unrolled_info, metric),
            atol=FLOAT_ATOL,
            rtol=FLOAT_RTOL,
        ), f"{metric} must match within strict tolerance"


def test_batched_chain_result_does_not_depend_on_its_batch() -> None:
    """A chain's transition must not depend on which chains it is batched with.

    ``vmap`` of ``lax.while_loop`` runs while any lane is still active and
    freezes finished lanes. This test pins that freezing down: a shallow chain
    batched with deep chains must get the same result as when it runs alone.
    The whole bucketing method rests on this, because bucketing does nothing
    but change which chains share a batch.
    """
    model = FunnelModel(dimension=5)
    num_chains = 8
    positions = jr.normal(jr.PRNGKey(11), (num_chains, 5), dtype=jnp.float32)
    keys = jr.split(jr.PRNGKey(12), num_chains)
    batched_state = new_multi_chain_state(model, positions)

    batched_next, batched_info, batched_keys = jit_monolithic_transition(
        model,
        batched_state,
        keys,
        step_size=MIXED_STEP_SIZE,
        max_tree_depth=6,
    )

    realized_depths = [int(depth) for depth in batched_info.realized_tree_depth]
    assert max(realized_depths) > min(realized_depths), (
        "test setup must produce heterogeneous realized depths, "
        f"got {realized_depths}"
    )

    for chain in range(num_chains):
        single_state = new_sampler_state(model, positions[chain])
        solo_next, solo_info, solo_key = one_chain_nuts_transition(
            model,
            single_state,
            keys[chain],
            step_size=MIXED_STEP_SIZE,
            max_tree_depth=6,
        )

        assert jnp.array_equal(solo_key, batched_keys[chain])
        for metric in EXACT_METRICS:
            assert jnp.array_equal(
                getattr(solo_info, metric),
                getattr(batched_info, metric)[chain],
            ), f"chain {chain}: {metric} changed under batching"
        assert jnp.allclose(
            solo_next.position,
            batched_next.position[chain],
            atol=FLOAT_ATOL,
            rtol=FLOAT_RTOL,
        )


def _warm_time(model: Any, state: Any, keys: Any, step_size: float, repeats: int = 5) -> Any:
    """Blocked warm time and realized depths for one batched transition."""

    def run() -> Any:
        return jit_monolithic_transition(
            model,
            state,
            keys,
            step_size=step_size,
            max_tree_depth=6,
        )

    outputs = block_until_ready_tree(run())  # compile before timing
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        outputs = block_until_ready_tree(run())
        timings.append(time.perf_counter() - start)
    return min(timings), outputs[1].realized_tree_depth


def test_warm_runtime_depends_on_realized_tree_depth() -> None:
    """Mechanism gate: shallow batches must be cheaper than deep ones.

    Before T39 this ratio was ~1.1 — a batch doing a quarter of the leapfrog
    work took the same wall time, because work was a function of
    ``max_tree_depth`` alone. That is precisely the condition under which no
    scheduler can win, so it is asserted here rather than left to a benchmark.

    Both cases share one compiled program (``step_size`` is a traced argument,
    only ``model`` and ``max_tree_depth`` are static), so this compares
    execution, not compilation.
    """
    model = FunnelModel(dimension=8)
    num_chains = 32
    positions = jr.normal(jr.PRNGKey(21), (num_chains, 8), dtype=jnp.float32)
    keys = jr.split(jr.PRNGKey(22), num_chains)
    state = new_multi_chain_state(model, positions)

    deep_time, deep_depths = _warm_time(model, state, keys, DEEP_STEP_SIZE)
    shallow_time, shallow_depths = _warm_time(model, state, keys, SHALLOW_STEP_SIZE)

    # Guard the setup: if these regimes stop differing, the timing assertion
    # below would pass vacuously.
    assert int(jnp.max(deep_depths)) >= 4, f"deep case must build deep trees, got {deep_depths}"
    assert int(jnp.max(shallow_depths)) == 1, (
        f"shallow case must stop at depth 1, got {shallow_depths}"
    )

    # Generous threshold: the mechanism gives a large margin, and this must not
    # be flaky on a loaded machine. The pre-T39 value was ~1.1.
    assert shallow_time < 0.6 * deep_time, (
        "warm runtime must depend on realized tree depth: "
        f"shallow={shallow_time:.6f}s deep={deep_time:.6f}s "
        f"ratio={shallow_time / deep_time:.3f} (expected < 0.6)"
    )
