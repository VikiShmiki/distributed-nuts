"""Does the sampler actually sample the right distribution?

Every other test in this suite checks *consistency*: monolithic against
bucketed, control-flow against unrolled, jitted against eager. All of those
compare two paths through the same transition, so a defect in the transition
itself passes them unanimously. That is not hypothetical — on 2026-08-09 the
U-turn criterion was found to be evaluated with its endpoints in visit order
rather than trajectory order, which made every backward subtree report a U-turn
immediately and turned realized tree depth into a coin-flip sequence. All 88
tests passed before and after the fix.

This file closes that gap with two independent kinds of check against an
isotropic Gaussian, whose moments are exact and which NUTS handles easily, so a
failure means the sampler is wrong rather than that the target is hard:

*Validity* — pooled draws must recover the analytic mean and standard deviation.

*Adaptivity* — realized tree depth must respond to the target's geometry. This
is the property that distinguishes NUTS from fixed-length HMC, and it is what
actually catches the 2026-08-09 defect.

Worth recording honestly: the moment checks alone do **not** catch that defect.
Measured against the pre-fix transition they pass, because the broken sampler is
not grossly biased — it degenerates into short one-sided HMC, which still
targets the right distribution on an easy target. It took 2.58 leapfrog steps
per transition against the fixed sampler's 10.04, with lag-1 autocorrelation
0.787 against 0.428. So it sampled acceptably and mixed badly. The adaptivity
check is what separates them: mean depth responds to a 10x change in target
scale by 1.85x when fixed and 1.08x when broken.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import jax.random as jr

from abnuts.models.base import ModelMetadata
from abnuts.nuts.monolithic import new_multi_chain_state, run_monolithic_jit

LOG_TWO_PI = math.log(2.0 * math.pi)


@dataclass(frozen=True)
class IsotropicGaussianModel:
    """Isotropic normal with standard deviation ``scale`` in ``dimension`` dims."""

    dimension: int
    scale: float = 1.0
    name: str = "isotropic_gaussian"

    @property
    def metadata(self) -> ModelMetadata:
        """Return serializable model metadata."""
        return ModelMetadata(
            name=self.name,
            dimension=self.dimension,
            event_shape=(self.dimension,),
            description="Isotropic standard normal test target with analytic moments.",
        )

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> Any:
        """Draw overdispersed starting points so the chains must actually move."""
        del config
        return jr.normal(jr.PRNGKey(key), (num_chains, self.dimension)) * 2.0 * self.scale

    def log_prob(self, position: Any, data: Any | None = None) -> Any:
        """Evaluate the standard normal log density."""
        del data
        x = jnp.asarray(position)
        return -0.5 * jnp.sum(x * x) / self.scale**2 - self.dimension * (
            math.log(self.scale) + 0.5 * LOG_TWO_PI
        )


def _sample(dimension: int, num_chains: int, num_steps: int, warmup: int) -> Any:
    model = IsotropicGaussianModel(dimension=dimension)
    positions = jnp.asarray(model.initial_position(key=0, num_chains=num_chains))
    result = run_monolithic_jit(
        model,
        positions,
        jr.PRNGKey(11),
        num_steps=num_steps,
        step_size=0.35,
        max_tree_depth=6,
    )
    # trace_positions is (steps, chains, dimension); drop warmup, pool the rest.
    return result.trace_positions[warmup:].reshape(-1, dimension), result


def test_sampler_recovers_isotropic_gaussian_moments() -> None:
    """Pooled draws must match the target's analytic mean and standard deviation.

    Tolerances are sized for Monte Carlo error at this sample size. See the
    module docstring for why this check is necessary but not sufficient.
    """
    dimension, num_chains, num_steps, warmup = 4, 256, 40, 20
    draws, _ = _sample(dimension, num_chains, num_steps, warmup)

    assert draws.shape[0] == num_chains * (num_steps - warmup)
    assert bool(jnp.all(jnp.isfinite(draws)))

    mean = jnp.mean(draws, axis=0)
    std = jnp.std(draws, axis=0)

    assert bool(jnp.all(jnp.abs(mean) < 0.15)), f"mean should be ~0, got {mean}"
    assert bool(jnp.all(jnp.abs(std - 1.0) < 0.15)), f"std should be ~1, got {std}"


def test_sampler_explores_rather_than_sticking() -> None:
    """Chains must move. A sampler that rejects everything also has mean ~0.

    The moment test alone can be satisfied by a chain that never moves from a
    symmetric starting distribution, so this pins down that the sampler is
    actually mixing.
    """
    dimension, num_chains, num_steps = 4, 64, 20
    _, result = _sample(dimension, num_chains, num_steps, warmup=0)
    trace = result.trace_positions

    moved = jnp.mean(jnp.any(trace[1:] != trace[:-1], axis=-1))
    assert float(moved) > 0.5, f"chains accepted a move on only {float(moved):.1%} of steps"

    # Starting scale is 2.0 against a target scale of 1.0, so the pooled spread
    # must contract toward the target rather than stay at the initial value.
    assert float(jnp.std(trace[-1])) < float(jnp.std(trace[0]))


def test_realized_depth_adapts_to_target_scale() -> None:
    """Trajectory length must be driven by geometry, not by coin flips.

    This is the statistical guard for the 2026-08-09 defect, and the one that
    actually discriminates. Holding the step size fixed and widening the target
    by 10x means a trajectory has further to travel before it U-turns, so
    realized depth must rise substantially. When the U-turn check was direction
    dependent, depth was geometric with p=0.5 on every target and barely moved:
    measured ratios were 1.85x fixed against 1.08x broken.

    Asserting a ratio rather than absolute depths keeps this robust to step-size
    and dimension choices.
    """
    from abnuts.nuts.monolithic import jit_monolithic_transition

    def mean_depth(scale: float) -> float:
        model = IsotropicGaussianModel(dimension=4, scale=scale)
        positions = jnp.asarray(model.initial_position(key=0, num_chains=512))
        state = new_multi_chain_state(model, positions)
        keys = jr.split(jr.PRNGKey(5), 512)
        _, info, _ = jit_monolithic_transition(
            model, state, keys, step_size=0.3, max_tree_depth=8
        )
        return float(jnp.mean(info.realized_tree_depth))

    narrow, wide = mean_depth(1.0), mean_depth(10.0)
    assert wide > narrow * 1.4, (
        "realized tree depth must respond to target geometry: "
        f"scale=1 gave {narrow:.2f}, scale=10 gave {wide:.2f} "
        f"(ratio {wide / narrow:.2f}x, expected > 1.4x)"
    )


def test_turning_criterion_is_symmetric_under_integration_direction() -> None:
    """A subtree's U-turn verdict must not depend on which way it was built.

    This is the direct regression guard for the 2026-08-09 defect. The criterion
    is a statement about geometry, so building the same span forwards or
    backwards has to agree.
    """
    from abnuts.nuts.integrator import leapfrog_step
    from abnuts.nuts.transition import (
        _integrator_from_sampler_state,
        _is_turning,
        _span_endpoints,
        new_sampler_state,
    )

    model = IsotropicGaussianModel(dimension=6)
    position = jr.normal(jr.PRNGKey(1), (6,)) * 0.5
    momentum = jr.normal(jr.PRNGKey(2), (6,))
    start = _integrator_from_sampler_state(new_sampler_state(model, position), momentum)

    for magnitude in (0.05, 0.2, 0.5):
        verdicts = []
        for backward in (False, True):
            step = -magnitude if backward else magnitude
            current = leapfrog_step(
                model, start, step_size=step, inverse_mass_matrix=None, data=None
            )
            left, right = _span_endpoints(jnp.asarray(backward), start, current)
            verdicts.append(bool(_is_turning(left, right, None)))
        assert verdicts[0] == verdicts[1], (
            f"step {magnitude}: forward said turning={verdicts[0]}, "
            f"backward said turning={verdicts[1]}"
        )


def test_subtree_uturn_matches_brute_force_recursion() -> None:
    """The checkpoint stack must reproduce the standard NUTS recursion exactly.

    The differential test between the control-flow and unrolled implementations
    only shows the two agree. This checks the criterion *itself* against a
    brute-force reference that stores every state in a subtree and tests every
    aligned power-of-two block directly, which is the definition the recursion
    unrolls to.

    The real builder stops at the first turning block, so the reference is
    compared on both the verdict and the step at which it fires.
    """
    import jax
    from abnuts.nuts.integrator import leapfrog_step
    from abnuts.nuts.transition import (
        _build_subtree_unrolled,
        _integrator_from_sampler_state,
        _is_turning_between,
        new_sampler_state,
    )

    model = IsotropicGaussianModel(dimension=5)

    def brute_force(start, step_size, num_steps):
        """First aligned block that U-turns, or None. Stores all states."""
        states = [start]
        for _ in range(num_steps):
            states.append(
                leapfrog_step(model, states[-1], step_size=step_size,
                              inverse_mass_matrix=None, data=None)
            )
        backward = jnp.asarray(step_size < 0)
        for step_index in range(num_steps):
            for level in range(1, max(1, num_steps.bit_length())):
                block = 1 << level
                if (step_index + 1) % block != 0 or block > num_steps:
                    continue
                left = states[step_index + 1 - block]
                right = states[step_index + 1]
                lp, lm = (right.position, right.momentum) if backward else (left.position, left.momentum)
                rp, rm = (left.position, left.momentum) if backward else (right.position, right.momentum)
                if bool(_is_turning_between(lp, lm, rp, rm, None)):
                    return step_index
        return None

    checked = 0
    for seed in range(4):
        for magnitude in (0.3, 0.8, 1.5):
            for backward in (False, True):
                for depth in (1, 2, 3):
                    num_steps = 1 << depth
                    step = -magnitude if backward else magnitude
                    position = jr.normal(jr.PRNGKey(seed), (5,)) * 0.7
                    momentum = jr.normal(jr.PRNGKey(seed + 50), (5,))
                    start = _integrator_from_sampler_state(
                        new_sampler_state(model, position), momentum
                    )
                    expected_step = brute_force(start, step, num_steps)
                    result = _build_subtree_unrolled(
                        model=model, start=start, rng_key=jr.PRNGKey(0),
                        log_slice=jnp.asarray(-1e30), initial_energy=jnp.asarray(0.0),
                        step_size=jnp.asarray(step), num_steps=num_steps,
                        active=jnp.asarray(True), integrate_backward=jnp.asarray(backward),
                        inverse_mass_matrix=None,
                        divergence_threshold=jnp.asarray(1e30), data=None,
                    )
                    assert bool(result.turning) == (expected_step is not None), (
                        f"seed={seed} step={step} depth={depth}: "
                        f"builder said turning={bool(result.turning)}, "
                        f"brute force said {expected_step is not None}"
                    )
                    if expected_step is not None:
                        assert int(result.leapfrog_count) == expected_step + 1, (
                            f"seed={seed} step={step} depth={depth}: builder stopped at "
                            f"{int(result.leapfrog_count)} steps, brute force at {expected_step + 1}"
                        )
                    checked += 1
    assert checked == 4 * 3 * 2 * 3
