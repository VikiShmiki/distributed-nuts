from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from abnuts.models.funnel import FunnelModel
from abnuts.nuts.hmc import run_fixed_hmc
from abnuts.nuts.independent import run_independent_chains_local
from abnuts.nuts.monolithic import run_monolithic


def test_fixed_hmc_runs_with_shared_leapfrog_schema() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=5, num_chains=6), dtype=jnp.float32)

    result = run_fixed_hmc(
        model,
        positions,
        jr.PRNGKey(13),
        num_steps=2,
        step_size=0.02,
        num_leapfrog_steps=3,
    )

    assert result.trace_positions.shape == (2, 6, 4)
    assert result.transition_info.acceptance_statistic.shape == (2, 6)
    assert jnp.all(result.transition_info.leapfrog_count == 3)
    assert jnp.all(result.transition_info.realized_tree_depth == 0)
    assert jnp.all(jnp.isfinite(result.final_state.position))


def test_independent_local_matches_monolithic_nuts_transition() -> None:
    model = FunnelModel(dimension=4)
    positions = jnp.asarray(model.initial_position(key=7, num_chains=5), dtype=jnp.float32)
    key = jr.PRNGKey(17)

    monolithic = run_monolithic(
        model,
        positions,
        key,
        num_steps=2,
        step_size=0.03,
        max_tree_depth=3,
    )
    independent = run_independent_chains_local(
        model,
        positions,
        key,
        num_steps=2,
        step_size=0.03,
        max_tree_depth=3,
    )

    assert independent.baseline_type == "independent_chain_local"
    assert jnp.array_equal(independent.trace_positions, monolithic.trace_positions)
    assert jnp.array_equal(independent.final_rng_keys, monolithic.final_rng_keys)
    for independent_metric, monolithic_metric in zip(
        independent.transition_info,
        monolithic.transition_info,
        strict=True,
    ):
        assert jnp.array_equal(independent_metric, monolithic_metric)
