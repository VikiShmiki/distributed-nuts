"""T40 gate: buckets must exit independently and execute less work.

The wall-clock question is hardware-dependent and belongs to T41. What is
checked here is the mechanism itself, which is not:

Structure
    The executor must scan the bucket axis, so each bucket gets its own
    ``lax.while_loop``. Flattening the bucket rectangle into a single
    transition, or vmapping over the bucket axis, would merge every bucket back
    under one ``any(active)`` predicate and give the reclaimed work straight
    back. That is exactly what the pre-T40 executor did.

Executed work
    Under ``vmap``, a group of chains costs its *slowest* member: the while loop
    runs until the deepest chain in the group stops, and every lane pays each
    iteration. So a group's executed lane-steps are ``len(group) * max leapfrog
    count in group``. Bucketing chains by predicted work should lower that total
    against one undifferentiated batch, and this test asserts it does.

That work model is an approximation in one direction only: it assumes the
deepest lane drives every subtree, so it slightly overstates both sides. It is
used for the *ratio*, where the bias largely cancels, and it never substitutes
for the wall-clock measurements in ``results/raw``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from abnuts.models.funnel import FunnelModel
from abnuts.nuts.bucketed import bucketed_transition
from abnuts.nuts.monolithic import monolithic_transition, new_multi_chain_state
from abnuts.nuts.planner import make_bucket_plan

STEP_SIZE = 0.03
MAX_TREE_DEPTH = 8


def _heterogeneous_batch(num_chains: int, dimension: int) -> Any:
    """Funnel chains spread along the scale coordinate, which drives depth."""
    model = FunnelModel(dimension=dimension)
    positions = jnp.asarray(
        model.initial_position(key=0, num_chains=num_chains),
        dtype=jnp.float32,
    )
    scale = jr.normal(jr.PRNGKey(7), (num_chains,), dtype=jnp.float32) * 2.5
    positions = positions.at[:, 0].set(scale)
    state = new_multi_chain_state(model, positions)
    keys = jr.split(jr.PRNGKey(3), num_chains)
    return model, state, keys


def _group_lane_steps(leapfrog_counts: Any, lane_indices: Any) -> int:
    """Executed lane-steps for one group: every lane pays the group maximum."""
    return int(lane_indices.shape[0]) * int(jnp.max(leapfrog_counts[lane_indices]))


def _count_primitive(jaxpr: Any, name: str) -> int:
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


def _plan_for(state: Any, keys: Any, model: Any, bucket_size: int) -> Any:
    """Oracle plan: bucket by the realized depth of a reference transition.

    Oracle grouping isolates the executor from predictor quality, which is what
    T40 is about. It is an analysis-only construction in the same sense as
    ``oracle_current`` in ``results/raw/oracle_gap`` and is not a deployable
    scheduler.
    """
    _, reference_info, _ = monolithic_transition(
        model,
        state,
        keys,
        step_size=STEP_SIZE,
        max_tree_depth=MAX_TREE_DEPTH,
    )
    return make_bucket_plan(
        reference_info.realized_tree_depth.astype(jnp.float32),
        canonical_bucket_sizes=bucket_size,
    )


def test_bucket_executor_scans_over_the_bucket_axis() -> None:
    """Structural gate: each bucket must keep its own control flow.

    Deterministic, so a regression to a flattened or vmapped executor fails
    here regardless of timing noise.
    """
    model, state, keys = _heterogeneous_batch(32, 16)
    plan = _plan_for(state, keys, model, bucket_size=8)

    def run(sampler_state: Any, rng_keys: Any) -> Any:
        return bucketed_transition(
            model,
            sampler_state,
            rng_keys,
            plan,
            step_size=STEP_SIZE,
            max_tree_depth=MAX_TREE_DEPTH,
        )

    jaxpr = jax.make_jaxpr(run)(state, keys).jaxpr

    assert plan.num_buckets > 1, "test setup must produce more than one bucket"
    assert _count_primitive(jaxpr, "scan") >= 1, (
        "the bucket executor must scan the bucket axis so each bucket keeps its "
        "own while_loop; a flattened or vmapped executor merges them"
    )
    assert _count_primitive(jaxpr, "while") >= 1


def test_executor_program_size_is_independent_of_bucket_count() -> None:
    """No host-side per-bucket dispatch: one compiled program regardless of buckets.

    This replaces a runtime criterion that could not do its job. The gate used to
    assert that warm runtime did not scale with bucket count, which was meant to
    detect the pre-repair Python loop over buckets. Measured eight times on
    identical code, that ratio ranged 1.106 to 1.532 against a 1.25 threshold, so
    it decided pass/fail on scheduler noise.

    The property actually worth asserting is structural and exactly checkable: a
    Python loop over buckets emits per-bucket equations, so program size grows
    with bucket count, while ``lax.map`` emits one scanned body whatever the
    bucket count. Holding total chains fixed and varying only how they are
    partitioned, the jaxpr equation count must not move.
    """
    model, state, keys = _heterogeneous_batch(64, 16)

    sizes = {}
    for bucket_size in (8, 16, 32):
        plan = _plan_for(state, keys, model, bucket_size=bucket_size)

        def run(sampler_state: Any, rng_keys: Any, p: Any = plan) -> Any:
            return bucketed_transition(
                model,
                sampler_state,
                rng_keys,
                p,
                step_size=STEP_SIZE,
                max_tree_depth=MAX_TREE_DEPTH,
            )

        jaxpr = jax.make_jaxpr(run)(state, keys).jaxpr
        sizes[plan.num_buckets] = _count_equations(jaxpr)

    assert len(sizes) >= 3, f"expected distinct bucket counts, got {sizes}"
    assert len(set(sizes.values())) == 1, (
        "compiled program size must not depend on bucket count; a growing size "
        f"means work is being emitted per bucket on the host: {sizes}"
    )


def _count_equations(jaxpr: Any) -> int:
    """Total equations in a jaxpr, descending into nested jaxprs."""
    total = 0
    for equation in jaxpr.eqns:
        total += 1
        for parameter in equation.params.values():
            closed = getattr(parameter, "jaxpr", parameter)
            if hasattr(closed, "eqns"):
                total += _count_equations(closed)
            elif isinstance(parameter, tuple | list):
                for item in parameter:
                    inner = getattr(item, "jaxpr", item)
                    if hasattr(inner, "eqns"):
                        total += _count_equations(inner)
    return total


def test_bucketed_executes_less_leapfrog_work_than_monolithic() -> None:
    """Mechanism gate: bucketing must lower executed lane-steps."""
    model, state, keys = _heterogeneous_batch(64, 32)
    plan = _plan_for(state, keys, model, bucket_size=16)

    _, monolithic_info, _ = monolithic_transition(
        model,
        state,
        keys,
        step_size=STEP_SIZE,
        max_tree_depth=MAX_TREE_DEPTH,
    )
    _, bucketed_info, _ = bucketed_transition(
        model,
        state,
        keys,
        plan,
        step_size=STEP_SIZE,
        max_tree_depth=MAX_TREE_DEPTH,
    )

    # The transition itself must be unchanged; only the grouping differs.
    assert jnp.array_equal(monolithic_info.leapfrog_count, bucketed_info.leapfrog_count)

    depths = monolithic_info.realized_tree_depth
    assert int(jnp.max(depths)) > int(jnp.min(depths)), (
        f"test setup must produce heterogeneous depths, got {depths}"
    )

    counts = monolithic_info.leapfrog_count
    monolithic_work = _group_lane_steps(counts, jnp.arange(counts.shape[0]))
    bucketed_work = sum(
        _group_lane_steps(counts, plan.idx[bucket]) for bucket in range(plan.num_buckets)
    )

    assert bucketed_work < monolithic_work, (
        "bucketed execution must run fewer lane-steps than one undifferentiated "
        f"batch: bucketed={bucketed_work} monolithic={monolithic_work}"
    )


def test_padded_lanes_cannot_deepen_a_bucket() -> None:
    """Padding must not extend the work a bucket pays for.

    The planner fills padded lanes with the last real chain index in the
    bucket, so a padded lane replays a chain that is already there and cannot
    become the bucket's new slowest member. If padding ever switched to a
    sentinel or zero state, a padded lane could dominate the bucket's exit
    condition and quietly inflate executed work.
    """
    model, state, keys = _heterogeneous_batch(20, 16)
    plan = _plan_for(state, keys, model, bucket_size=8)

    _, info, _ = monolithic_transition(
        model,
        state,
        keys,
        step_size=STEP_SIZE,
        max_tree_depth=MAX_TREE_DEPTH,
    )
    counts = info.leapfrog_count

    assert int(jnp.sum(plan.padding_count)) > 0, "test setup must produce padding"

    for bucket in range(plan.num_buckets):
        lanes = plan.idx[bucket]
        mask = plan.mask[bucket]
        real_lanes = lanes[mask]
        assert int(jnp.max(counts[lanes])) == int(jnp.max(counts[real_lanes])), (
            f"bucket {bucket}: padded lanes changed the bucket maximum"
        )
