"""Run a tiny oracle-gap decomposition for bucketed NUTS schedulers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from functools import partial
from io import StringIO
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from abnuts.blocking import block_until_ready_tree
from abnuts.config import ConfigError, load_yaml_config
from abnuts.io import build_result_manifest
from abnuts.models import get_model
from abnuts.models.base import BenchmarkModel
from abnuts.nuts.bucketed import bucketed_transition
from abnuts.nuts.monolithic import new_multi_chain_state, run_monolithic_jit
from abnuts.nuts.planner import BucketPlan, make_bucket_plan
from abnuts.nuts.predictors import (
    PredictorState,
    hvp_curvature_work,
    new_predictor_state,
    predict_work,
    predictor_uses_hvp,
    update_hvp_work,
    update_predictor_state,
)
from abnuts.nuts.state import SamplerState
from abnuts.nuts.transition import TransitionInfo
from abnuts.profiling import TimingBreakdown, TimingRecorder

SCHEDULER_MODES = frozenset(
    {
        "monolithic",
        "unsorted",
        "random",
        "history",
        "hvp",
        "hybrid",
        "oracle_previous",
        "oracle_current",
    }
)
ANALYSIS_ONLY_MODES = frozenset({"oracle_current"})

# Warm wall time is the minimum of this many blocked repeats. A single shot is
# not reproducible; see _repeated_blocked_call.
DEFAULT_WARM_REPEATS = 3


class OracleGapConfig(NamedTuple):
    """Resolved oracle-gap configuration for one profile."""

    benchmark: str
    model: str
    backend: str
    dtype: str
    seeds: tuple[int, ...]
    num_chains: int
    dimension: int
    num_steps: int
    step_size: float
    max_tree_depth: int
    bucket_sizes: tuple[int, ...]
    scheduler_modes: tuple[str, ...]
    predictor_beta: float


class TimedResult(NamedTuple):
    """A run result plus elapsed blocked wall time in seconds."""

    result: Any
    elapsed_seconds: float
    timing: TimingBreakdown


class OracleBucketedRunResult(NamedTuple):
    """Outputs needed to summarize one oracle-gap bucketed scheduler."""

    initial_state: SamplerState
    final_state: SamplerState
    initial_rng_keys: jax.Array
    final_rng_keys: jax.Array
    trace_positions: jax.Array
    transition_info: TransitionInfo
    predictor_state: PredictorState
    bucket_plans: tuple[BucketPlan, ...]
    predicted_work: jax.Array
    hvp_overhead_seconds: float
    hvp_call_count: int
    timing: TimingBreakdown = TimingBreakdown()


class OracleGapOutputs(NamedTuple):
    """All CSV row groups emitted by the oracle-gap experiment."""

    summary_rows: list[dict[str, Any]]
    predictor_calibration_rows: list[dict[str, Any]]
    padding_heatmap_rows: list[dict[str, Any]]
    speedup_heterogeneity_rows: list[dict[str, Any]]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the oracle-gap runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], help="Override configured backend.")
    parser.add_argument("--profile", default="tiny")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_config(
    raw_config: dict[str, Any],
    *,
    profile: str,
    backend_override: str | None = None,
) -> OracleGapConfig:
    """Resolve top-level defaults plus one named profile into a typed config."""
    profiles = raw_config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("config field 'profiles' must be a mapping")
    if profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(
            f"profile {profile!r} not found in config. Available profiles: {available}"
        )
    profile_config = profiles[profile]
    if not isinstance(profile_config, dict):
        raise ValueError(f"profile {profile!r} must contain a mapping")

    merged = {key: value for key, value in raw_config.items() if key != "profiles"}
    merged.update(profile_config)
    if backend_override is not None:
        merged["backend"] = backend_override

    config = OracleGapConfig(
        benchmark=str(merged.get("benchmark", "oracle_gap")),
        model=str(merged.get("model", "funnel")),
        backend=str(merged.get("backend", "cpu")),
        dtype=str(merged.get("dtype", "float32")),
        seeds=_as_int_tuple(merged.get("seeds", (merged.get("seed", 0),)), "seeds"),
        num_chains=_as_positive_int(merged.get("num_chains", 4), "num_chains"),
        dimension=_as_positive_int(merged.get("dimension", 4), "dimension"),
        num_steps=_as_positive_int(merged.get("num_steps", 2), "num_steps"),
        step_size=_as_positive_float(merged.get("step_size", 0.03), "step_size"),
        max_tree_depth=_as_positive_int(merged.get("max_tree_depth", 2), "max_tree_depth"),
        bucket_sizes=_as_int_tuple(merged.get("bucket_sizes", (2,)), "bucket_sizes"),
        scheduler_modes=_as_str_tuple(
            merged.get("scheduler_modes", tuple(sorted(SCHEDULER_MODES))),
            "scheduler_modes",
        ),
        predictor_beta=_as_probability(merged.get("predictor_beta", 0.9), "predictor_beta"),
    )
    validate_config(config)
    return config


def validate_config(config: OracleGapConfig) -> None:
    """Validate resolved oracle-gap settings."""
    if config.benchmark != "oracle_gap":
        raise ValueError(f"expected benchmark='oracle_gap', got {config.benchmark!r}")
    if config.backend not in {"cpu", "gpu"}:
        raise ValueError(f"run_oracle_gap supports backend='cpu' or 'gpu', got {config.backend!r}")
    if config.dtype != "float32":
        raise ValueError(f"run_oracle_gap currently supports dtype='float32', got {config.dtype!r}")
    if not config.seeds:
        raise ValueError("seeds must contain at least one seed")
    if any(bucket_size <= 0 for bucket_size in config.bucket_sizes):
        raise ValueError(f"bucket_sizes must all be positive, got {config.bucket_sizes!r}")
    unknown_modes = sorted(set(config.scheduler_modes) - SCHEDULER_MODES)
    if unknown_modes:
        valid = ", ".join(sorted(SCHEDULER_MODES))
        raise ValueError(f"unsupported scheduler modes {unknown_modes!r}; expected one of {valid}")


def run_oracle_gap(config: OracleGapConfig) -> OracleGapOutputs:
    """Run all configured scheduler modes and return machine-readable rows."""
    summary_rows: list[dict[str, Any]] = []
    predictor_calibration_rows: list[dict[str, Any]] = []
    padding_heatmap_rows: list[dict[str, Any]] = []
    speedup_heterogeneity_rows: list[dict[str, Any]] = []
    model = get_model(config.model, dimension=config.dimension)
    for seed in config.seeds:
        initial_positions = jnp.asarray(
            model.initial_position(key=seed, num_chains=config.num_chains),
            dtype=jnp.float32,
        )
        block_until_ready_tree(initial_positions)
        mono_timed = _time_monolithic(model, initial_positions, config, seed=seed)
        mono_summary = _summarize_monolithic_row(config, seed, mono_timed)
        mono_realized = np.asarray(
            mono_timed.result.transition_info.realized_tree_depth,
            dtype=float,
        )
        if "monolithic" in config.scheduler_modes:
            summary_rows.append(mono_summary)
            speedup_heterogeneity_rows.append(
                _speedup_heterogeneity_row(
                    config,
                    seed,
                    scheduler_mode="monolithic",
                    scheduler_label="monolithic",
                    bucket_size=0,
                    is_analysis_upper_bound=False,
                    speedup=1.0,
                    realized_work=mono_realized,
                    bucket_max_over_global_max=1.0,
                )
            )

        current_oracle_work = mono_timed.result.transition_info.realized_tree_depth.astype(
            jnp.float32
        )
        for mode in config.scheduler_modes:
            if mode == "monolithic":
                continue
            for bucket_size in config.bucket_sizes:
                bucket_timed = _time_oracle_bucketed(
                    model,
                    initial_positions,
                    config,
                    seed=seed,
                    scheduler_mode=mode,
                    bucket_size=bucket_size,
                    current_oracle_work=current_oracle_work,
                )
                bucket_summary = _summarize_bucketed_row(
                    config,
                    seed,
                    scheduler_mode=mode,
                    bucket_size=bucket_size,
                    mono_timed=mono_timed,
                    bucket_timed=bucket_timed,
                )
                summary_rows.append(bucket_summary)
                predictor_calibration_rows.extend(
                    _predictor_calibration_rows(
                        config,
                        seed,
                        scheduler_mode=mode,
                        bucket_size=bucket_size,
                        result=bucket_timed.result,
                    )
                )
                padding_heatmap_rows.extend(
                    _padding_heatmap_rows(
                        config,
                        seed,
                        scheduler_mode=mode,
                        bucket_size=bucket_size,
                        result=bucket_timed.result,
                    )
                )
                speedup_heterogeneity_rows.append(
                    _speedup_heterogeneity_row(
                        config,
                        seed,
                        scheduler_mode=mode,
                        scheduler_label=_scheduler_label(mode),
                        bucket_size=bucket_size,
                        is_analysis_upper_bound=mode in ANALYSIS_ONLY_MODES,
                        speedup=float(bucket_summary["speedup"]),
                        realized_work=np.asarray(
                            bucket_timed.result.transition_info.realized_tree_depth,
                            dtype=float,
                        ),
                        bucket_max_over_global_max=float(
                            bucket_summary["bucket_max_over_global_max"]
                        ),
                    )
                )
    return OracleGapOutputs(
        summary_rows=summary_rows,
        predictor_calibration_rows=predictor_calibration_rows,
        padding_heatmap_rows=padding_heatmap_rows,
        speedup_heterogeneity_rows=speedup_heterogeneity_rows,
    )


def run_oracle_bucketed(
    model: BenchmarkModel,
    initial_positions: jax.Array,
    rng_key: jax.Array,
    *,
    num_steps: int,
    step_size: float,
    max_tree_depth: int,
    bucket_size: int | Sequence[int],
    scheduler_mode: str,
    predictor_beta: float,
    current_oracle_work: jax.Array,
    dtype: Any = jnp.float32,
    enable_timing_breakdown: bool = False,
    enable_profiler_markers: bool = False,
) -> OracleBucketedRunResult:
    """Run bucketed NUTS with oracle-gap scheduler controls.

    ``oracle_current`` uses precomputed same-iteration realized work and is an
    analysis-only upper bound, not a deployable scheduler.
    """
    if scheduler_mode not in SCHEDULER_MODES or scheduler_mode == "monolithic":
        raise ValueError(f"unsupported bucketed scheduler mode {scheduler_mode!r}")
    if scheduler_mode == "oracle_current" and current_oracle_work.shape[0] != num_steps:
        raise ValueError(
            "current_oracle_work must have shape (num_steps, num_chains) "
            f"for oracle_current; got {current_oracle_work.shape}"
        )

    positions = jnp.asarray(initial_positions, dtype=dtype)
    state = new_multi_chain_state(model, positions)
    initial_state = state
    initial_rng_keys = jr.split(rng_key, positions.shape[0])
    rng_keys = initial_rng_keys
    predictor_state = new_predictor_state(positions.shape[0], dtype=dtype)
    predictor_key = rng_key

    trace_positions: list[jax.Array] = []
    infos: list[TransitionInfo] = []
    plans: list[BucketPlan] = []
    predictions: list[jax.Array] = []
    hvp_overhead_seconds = 0.0
    hvp_call_count = 0
    timing_recorder = TimingRecorder(
        enabled=enable_timing_breakdown,
        enable_profiler_markers=enable_profiler_markers,
    )

    for step in range(num_steps):
        predictor_key, prediction_key = jr.split(predictor_key)
        if scheduler_mode in {"hvp", "hybrid"} and predictor_uses_hvp(scheduler_mode):
            if timing_recorder.enabled:
                hvp_work, hvp_elapsed = timing_recorder.timed_call(
                    "hvp",
                    partial(hvp_curvature_work, model, state.position),
                    block_before=state,
                    marker_name="abnuts:oracle_hvp_probe",
                )
                hvp_overhead_seconds += hvp_elapsed
            else:
                state = block_until_ready_tree(state)
                hvp_start = time.perf_counter()
                hvp_work = hvp_curvature_work(model, state.position)
                hvp_work = block_until_ready_tree(hvp_work)
                hvp_overhead_seconds += time.perf_counter() - hvp_start
            hvp_call_count += 1
            predictor_state = update_hvp_work(predictor_state, hvp_work)

        (predicted_work, plan), _ = timing_recorder.timed_call(
            "planner",
            partial(
                _predict_scheduler_work_and_plan,
                scheduler_mode,
                predictor_state,
                step=step,
                rng_key=prediction_key,
                current_oracle_work=current_oracle_work,
                bucket_size=bucket_size,
            ),
            block_before=predictor_state,
            marker_name="abnuts:oracle_bucket_planner",
        )
        state, info, rng_keys = bucketed_transition(
            model,
            state,
            rng_keys,
            plan,
            step_size=step_size,
            max_tree_depth=max_tree_depth,
            timing_recorder=timing_recorder,
        )
        predictor_state = update_predictor_state(
            predictor_state,
            info.realized_tree_depth.astype(dtype),
            beta=predictor_beta,
        )
        trace_positions.append(state.position)
        infos.append(info)
        plans.append(plan)
        predictions.append(jnp.asarray(predicted_work, dtype=dtype))

    stacked_info = jax.tree_util.tree_map(lambda *items: jnp.stack(items), *infos)
    return OracleBucketedRunResult(
        initial_state=initial_state,
        final_state=state,
        initial_rng_keys=initial_rng_keys,
        final_rng_keys=rng_keys,
        trace_positions=jnp.stack(trace_positions),
        transition_info=stacked_info,
        predictor_state=predictor_state,
        bucket_plans=tuple(plans),
        predicted_work=jnp.stack(predictions),
        hvp_overhead_seconds=hvp_overhead_seconds,
        hvp_call_count=hvp_call_count,
        timing=timing_recorder.snapshot(),
    )


def write_outputs(
    out_dir: Path,
    *,
    config_path: Path,
    profile: str,
    config: OracleGapConfig,
    outputs: OracleGapOutputs,
    overwrite: bool,
) -> None:
    """Write oracle-gap manifest and summary CSV."""
    _prepare_output_dir(out_dir, overwrite=overwrite)
    summary_csv = _csv_from_rows(outputs.summary_rows)
    predictor_calibration_csv = _csv_from_rows(outputs.predictor_calibration_rows)
    padding_heatmap_csv = _csv_from_rows(outputs.padding_heatmap_rows)
    speedup_heterogeneity_csv = _csv_from_rows(outputs.speedup_heterogeneity_rows)
    resolved_config = _config_to_json(config, profile=profile)
    manifest = build_result_manifest(
        command="python -m abnuts.experiments.run_oracle_gap",
        config_path=config_path,
        output_dir=out_dir,
        config=resolved_config,
        extra={
            "profile": profile,
            "benchmark": config.benchmark,
            "timing_breakdown_enabled": True,
            "row_count": len(outputs.summary_rows),
            "analysis_upper_bound_modes": sorted(ANALYSIS_ONLY_MODES),
            "outputs": {
                "summary_csv": "summary.csv",
                "predictor_calibration_csv": "predictor_calibration.csv",
                "padding_heatmap_csv": "padding_heatmap.csv",
                "speedup_heterogeneity_csv": "speedup_heterogeneity.csv",
            },
        },
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.csv").write_text(summary_csv, encoding="utf-8")
    (out_dir / "predictor_calibration.csv").write_text(
        predictor_calibration_csv,
        encoding="utf-8",
    )
    (out_dir / "padding_heatmap.csv").write_text(padding_heatmap_csv, encoding="utf-8")
    (out_dir / "speedup_heterogeneity.csv").write_text(
        speedup_heterogeneity_csv,
        encoding="utf-8",
    )


def _time_monolithic(
    model: BenchmarkModel,
    initial_positions: jax.Array,
    config: OracleGapConfig,
    *,
    seed: int,
) -> TimedResult:
    def run_once() -> Any:
        return run_monolithic_jit(
            model,
            initial_positions,
            jr.PRNGKey(seed),
            num_steps=config.num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            dtype=jnp.float32,
        )

    return _time_runner(
        run_once,
        enable_timing_breakdown=True,
        timing_from_result=None,
    )


def _time_oracle_bucketed(
    model: BenchmarkModel,
    initial_positions: jax.Array,
    config: OracleGapConfig,
    *,
    seed: int,
    scheduler_mode: str,
    bucket_size: int,
    current_oracle_work: jax.Array,
) -> TimedResult:
    def run_once() -> OracleBucketedRunResult:
        return run_oracle_bucketed(
            model,
            initial_positions,
            jr.PRNGKey(seed),
            num_steps=config.num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            bucket_size=bucket_size,
            scheduler_mode=scheduler_mode,
            predictor_beta=config.predictor_beta,
            current_oracle_work=current_oracle_work,
            dtype=jnp.float32,
            enable_timing_breakdown=True,
        )

    return _time_runner(
        run_once,
        enable_timing_breakdown=True,
        timing_from_result=lambda result: result.timing,
    )


def _time_runner(
    runner: Callable[[], Any],
    *,
    enable_timing_breakdown: bool,
    timing_from_result: Callable[[Any], TimingBreakdown] | None,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
) -> TimedResult:
    if not enable_timing_breakdown:
        result, elapsed_seconds = _repeated_blocked_call(runner, warm_repeats)
        return TimedResult(
            result=result,
            elapsed_seconds=elapsed_seconds,
            timing=TimingBreakdown(),
        )

    _cold_result, cold_run_seconds = _elapsed_blocked_call(runner)
    warm_result, warm_iteration_seconds = _repeated_blocked_call(runner, warm_repeats)
    if timing_from_result is None:
        timing = TimingBreakdown(
            enabled=True,
            executor_seconds=warm_iteration_seconds,
            total_profiled_seconds=warm_iteration_seconds,
        )
    else:
        timing = timing_from_result(warm_result)
    timing = timing.with_outer_timings(
        cold_run_seconds=cold_run_seconds,
        warm_iteration_seconds=warm_iteration_seconds,
    )
    return TimedResult(
        result=warm_result,
        elapsed_seconds=warm_iteration_seconds,
        timing=timing,
    )


def _repeated_blocked_call(runner: Callable[[], Any], repeats: int) -> tuple[Any, float]:
    """Run a blocked call ``repeats`` times and keep the fastest.

    A single warm measurement is not reproducible on a shared node. See the
    matching helper in ``run_benchmark`` for the measured evidence.
    """
    if repeats <= 0:
        raise ValueError(f"warm_repeats must be positive, got {repeats!r}")
    best_seconds = math.inf
    result = None
    for _ in range(repeats):
        result, elapsed_seconds = _elapsed_blocked_call(runner)
        best_seconds = min(best_seconds, elapsed_seconds)
    return result, best_seconds


def _elapsed_blocked_call(runner: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = runner()
    result = block_until_ready_tree(result)
    stop = time.perf_counter()
    return result, stop - start


def _predict_scheduler_work(
    scheduler_mode: str,
    predictor_state: PredictorState,
    *,
    step: int,
    rng_key: jax.Array,
    current_oracle_work: jax.Array,
) -> jax.Array:
    if scheduler_mode == "unsorted":
        return jnp.ones_like(predictor_state.history_ema_work)
    if scheduler_mode == "oracle_previous":
        if step == 0:
            return jnp.ones_like(predictor_state.history_ema_work)
        return predictor_state.last_realized_work
    if scheduler_mode == "oracle_current":
        return current_oracle_work[step]
    if scheduler_mode == "random":
        return predict_work("random", predictor_state, rng_key=rng_key)
    return predict_work(scheduler_mode, predictor_state)


def _predict_scheduler_work_and_plan(
    scheduler_mode: str,
    predictor_state: PredictorState,
    *,
    step: int,
    rng_key: jax.Array,
    current_oracle_work: jax.Array,
    bucket_size: int | Sequence[int],
) -> tuple[jax.Array, BucketPlan]:
    predicted_work = _predict_scheduler_work(
        scheduler_mode,
        predictor_state,
        step=step,
        rng_key=rng_key,
        current_oracle_work=current_oracle_work,
    )
    plan = make_bucket_plan(predicted_work, canonical_bucket_sizes=bucket_size)
    return predicted_work, plan


def _summarize_monolithic_row(
    config: OracleGapConfig,
    seed: int,
    mono_timed: TimedResult,
) -> dict[str, Any]:
    result = mono_timed.result
    info = result.transition_info
    realized = np.asarray(info.realized_tree_depth, dtype=float)
    leapfrog_counts = np.asarray(info.leapfrog_count)
    executed_lane_steps = _monolithic_executed_lane_steps(leapfrog_counts, config.num_chains)
    row = {
        "benchmark": config.benchmark,
        "model": config.model,
        "seed": seed,
        "scheduler_mode": "monolithic",
        "scheduler_label": "monolithic",
        "is_analysis_upper_bound": False,
        "bucket_size": 0,
        "num_chains": config.num_chains,
        "dimension": config.dimension,
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "predictor_beta": config.predictor_beta,
        "t_mono": mono_timed.elapsed_seconds,
        "t_mode": mono_timed.elapsed_seconds,
        "speedup": 1.0,
        "padding_ratio": 0.0,
        "padding_count": 0,
        "mean_predictor_abs_error": 0.0,
        "mean_bucket_realized_max": float(np.mean(np.max(realized, axis=1))),
        "mean_sum_bucket_realized_max": float(np.mean(np.max(realized, axis=1))),
        "mean_global_realized_max": float(np.mean(np.max(realized, axis=1))),
        "bucket_max_over_global_max": 1.0,
        "total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
        "divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
        "max_realized_tree_depth": int(np.max(realized)),
        # A monolithic run is one undifferentiated group, so it is its own
        # reference and its own best case.
        "executed_lane_steps": executed_lane_steps,
        "monolithic_executed_lane_steps": executed_lane_steps,
        "executed_work_ratio": 1.0,
        "oracle_plan_executed_lane_steps": executed_lane_steps,
        "oracle_plan_executed_work_ratio": 1.0,
        "hvp_overhead_seconds": 0.0,
        "hvp_call_count": 0,
    }
    row.update(_timing_summary_fields(mono_timed))
    return row


def _summarize_bucketed_row(
    config: OracleGapConfig,
    seed: int,
    *,
    scheduler_mode: str,
    bucket_size: int,
    mono_timed: TimedResult,
    bucket_timed: TimedResult,
) -> dict[str, Any]:
    result = bucket_timed.result
    info = result.transition_info
    realized = np.asarray(info.realized_tree_depth, dtype=float)
    predicted = np.asarray(result.predicted_work, dtype=float)
    plan_summary = _summarize_plans(result.bucket_plans, realized)
    speedup = mono_timed.elapsed_seconds / bucket_timed.elapsed_seconds

    # Executed work is measured against this mode's own realized leapfrog
    # counts. Grouping does not change the transition, so the same counts are
    # the correct reference for the monolithic and oracle-plan comparisons.
    leapfrog_counts = np.asarray(info.leapfrog_count)
    executed_lane_steps = _plan_executed_lane_steps(leapfrog_counts, result.bucket_plans)
    monolithic_lane_steps = _monolithic_executed_lane_steps(leapfrog_counts, config.num_chains)
    oracle_lane_steps = _oracle_plan_executed_lane_steps(
        leapfrog_counts,
        realized,
        bucket_size,
    )
    row = {
        "benchmark": config.benchmark,
        "model": config.model,
        "seed": seed,
        "scheduler_mode": scheduler_mode,
        "scheduler_label": _scheduler_label(scheduler_mode),
        "is_analysis_upper_bound": scheduler_mode in ANALYSIS_ONLY_MODES,
        "bucket_size": bucket_size,
        "num_chains": config.num_chains,
        "dimension": config.dimension,
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "predictor_beta": config.predictor_beta,
        "t_mono": mono_timed.elapsed_seconds,
        "t_mode": bucket_timed.elapsed_seconds,
        "speedup": speedup,
        "padding_ratio": plan_summary["padding_ratio"],
        "padding_count": plan_summary["padding_count"],
        "mean_predictor_abs_error": float(np.mean(np.abs(predicted - realized))),
        "mean_bucket_realized_max": plan_summary["mean_bucket_realized_max"],
        "mean_sum_bucket_realized_max": plan_summary["mean_sum_bucket_realized_max"],
        "mean_global_realized_max": plan_summary["mean_global_realized_max"],
        "bucket_max_over_global_max": plan_summary["bucket_max_over_global_max"],
        "total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
        "divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
        "max_realized_tree_depth": int(np.max(realized)),
        "executed_lane_steps": executed_lane_steps,
        "monolithic_executed_lane_steps": monolithic_lane_steps,
        "executed_work_ratio": executed_lane_steps / monolithic_lane_steps,
        "oracle_plan_executed_lane_steps": oracle_lane_steps,
        "oracle_plan_executed_work_ratio": oracle_lane_steps / monolithic_lane_steps,
        "hvp_overhead_seconds": float(result.hvp_overhead_seconds),
        "hvp_call_count": int(result.hvp_call_count),
    }
    row.update(_timing_summary_fields(bucket_timed))
    return row


def _monolithic_executed_lane_steps(leapfrog_counts: np.ndarray, num_chains: int) -> int:
    """Lane-steps one undifferentiated batch executes, summed over steps.

    Under vmap a group costs its slowest member: the while loop runs until the
    deepest chain stops and every lane pays each iteration. See the executed-work
    model note in ``tests/test_bucketed_executor_work.py``.
    """
    return int(num_chains * np.sum(np.max(leapfrog_counts, axis=1)))


def _plan_executed_lane_steps(leapfrog_counts: np.ndarray, plans: Sequence[BucketPlan]) -> int:
    """Lane-steps a bucket schedule executes, summed over steps and buckets."""
    total = 0
    for step, plan in enumerate(plans):
        idx = np.asarray(plan.idx)
        lane_counts = leapfrog_counts[step][idx]
        total += int(idx.shape[1]) * int(np.sum(np.max(lane_counts, axis=1)))
    return total


def _oracle_plan_executed_lane_steps(
    leapfrog_counts: np.ndarray,
    realized: np.ndarray,
    bucket_size: int,
) -> int:
    """Lane-steps an oracle plan would execute on the same realized depths.

    Compared against the deployed scheduler's plans, this separates predictor
    quality from executor structure: the executor is identical, only the
    grouping differs. Analysis-only, in the same sense as ``oracle_current``.
    """
    plans = [
        make_bucket_plan(
            jnp.asarray(realized[step], dtype=jnp.float32),
            canonical_bucket_sizes=bucket_size,
        )
        for step in range(realized.shape[0])
    ]
    return _plan_executed_lane_steps(leapfrog_counts, plans)


def _timing_summary_fields(timed: TimedResult) -> dict[str, Any]:
    fields = dict(timed.timing.as_summary_fields(elapsed_seconds=timed.elapsed_seconds))
    gather_scatter_seconds = float(fields["timing_gather_seconds"]) + float(
        fields["timing_scatter_seconds"]
    )
    non_executor_overhead_seconds = (
        float(fields["timing_planner_seconds"])
        + gather_scatter_seconds
        + float(fields["timing_hvp_seconds"])
        + float(fields["timing_unattributed_seconds"])
    )
    components = {
        "planner": float(fields["timing_planner_seconds"]),
        "executor": float(fields["timing_executor_seconds"]),
        "gather_scatter": gather_scatter_seconds,
        "hvp": float(fields["timing_hvp_seconds"]),
        "unattributed": float(fields["timing_unattributed_seconds"]),
    }
    dominant_component, dominant_seconds = max(
        components.items(),
        key=lambda item: item[1],
    )
    fields.update(
        {
            "timing_gather_scatter_seconds": gather_scatter_seconds,
            "timing_non_executor_overhead_seconds": non_executor_overhead_seconds,
            "timing_dominant_warm_component": dominant_component,
            "timing_dominant_warm_component_seconds": dominant_seconds,
        }
    )
    return fields


def _summarize_plans(
    plans: tuple[BucketPlan, ...],
    realized_work: np.ndarray,
) -> dict[str, float | int]:
    total_padding = 0
    total_lanes = 0
    bucket_maxima: list[float] = []
    sum_bucket_maxima: list[float] = []
    global_maxima: list[float] = []

    for step, plan in enumerate(plans):
        idx = np.asarray(plan.idx)
        mask = np.asarray(plan.mask, dtype=bool)
        bucket_sizes = np.asarray(plan.bucket_sizes, dtype=int)
        occupancy = np.asarray(plan.occupancy, dtype=int)
        total_padding += int(np.sum(bucket_sizes - occupancy))
        total_lanes += int(np.sum(bucket_sizes))

        step_bucket_maxima: list[float] = []
        for bucket_number in range(plan.num_buckets):
            real_idx = idx[bucket_number][mask[bucket_number]]
            maximum = float(np.max(realized_work[step, real_idx]))
            step_bucket_maxima.append(maximum)
            bucket_maxima.append(maximum)
        sum_bucket_maxima.append(float(np.sum(step_bucket_maxima)))
        global_maxima.append(float(np.max(realized_work[step])))

    mean_global = float(np.mean(global_maxima))
    mean_sum_bucket = float(np.mean(sum_bucket_maxima))
    return {
        "padding_count": total_padding,
        "padding_ratio": float(total_padding / total_lanes) if total_lanes else 0.0,
        "mean_bucket_realized_max": float(np.mean(bucket_maxima)),
        "mean_sum_bucket_realized_max": mean_sum_bucket,
        "mean_global_realized_max": mean_global,
        "bucket_max_over_global_max": mean_sum_bucket / mean_global if mean_global else 0.0,
    }


def _predictor_calibration_rows(
    config: OracleGapConfig,
    seed: int,
    *,
    scheduler_mode: str,
    bucket_size: int,
    result: OracleBucketedRunResult,
) -> list[dict[str, Any]]:
    predicted = np.asarray(result.predicted_work, dtype=float)
    realized = np.asarray(result.transition_info.realized_tree_depth, dtype=float)
    rows: list[dict[str, Any]] = []
    for step in range(predicted.shape[0]):
        for chain in range(predicted.shape[1]):
            rows.append(
                {
                    "benchmark": config.benchmark,
                    "model": config.model,
                    "seed": seed,
                    "scheduler_mode": scheduler_mode,
                    "scheduler_label": _scheduler_label(scheduler_mode),
                    "is_analysis_upper_bound": scheduler_mode in ANALYSIS_ONLY_MODES,
                    "bucket_size": bucket_size,
                    "step": step,
                    "chain": chain,
                    "predicted_work": float(predicted[step, chain]),
                    "realized_work": float(realized[step, chain]),
                    "abs_error": float(abs(predicted[step, chain] - realized[step, chain])),
                    "is_summary_proxy": False,
                }
            )
    return rows


def _padding_heatmap_rows(
    config: OracleGapConfig,
    seed: int,
    *,
    scheduler_mode: str,
    bucket_size: int,
    result: OracleBucketedRunResult,
) -> list[dict[str, Any]]:
    realized = np.asarray(result.transition_info.realized_tree_depth, dtype=float)
    rows: list[dict[str, Any]] = []
    for step, plan in enumerate(result.bucket_plans):
        idx = np.asarray(plan.idx)
        mask = np.asarray(plan.mask, dtype=bool)
        bucket_sizes = np.asarray(plan.bucket_sizes, dtype=int)
        occupancy = np.asarray(plan.occupancy, dtype=int)
        bucket_padding = np.asarray(plan.bucket_padding_count, dtype=int)
        bucket_min = np.asarray(plan.bucket_min_predicted_work, dtype=float)
        bucket_max = np.asarray(plan.bucket_max_predicted_work, dtype=float)
        global_realized_max = float(np.max(realized[step]))
        for bucket_number in range(plan.num_buckets):
            real_idx = idx[bucket_number][mask[bucket_number]]
            rows.append(
                {
                    "benchmark": config.benchmark,
                    "model": config.model,
                    "seed": seed,
                    "scheduler_mode": scheduler_mode,
                    "scheduler_label": _scheduler_label(scheduler_mode),
                    "is_analysis_upper_bound": scheduler_mode in ANALYSIS_ONLY_MODES,
                    "bucket_size": bucket_size,
                    "step": step,
                    "bucket_index": bucket_number,
                    "bucket_capacity": int(bucket_sizes[bucket_number]),
                    "occupancy": int(occupancy[bucket_number]),
                    "padding_count": int(bucket_padding[bucket_number]),
                    "fill_fraction": float(
                        occupancy[bucket_number] / bucket_sizes[bucket_number]
                    ),
                    "bucket_min_predicted_work": float(bucket_min[bucket_number]),
                    "bucket_max_predicted_work": float(bucket_max[bucket_number]),
                    "bucket_realized_max": float(np.max(realized[step, real_idx])),
                    "global_realized_max": global_realized_max,
                }
            )
    return rows


def _speedup_heterogeneity_row(
    config: OracleGapConfig,
    seed: int,
    *,
    scheduler_mode: str,
    scheduler_label: str,
    bucket_size: int,
    is_analysis_upper_bound: bool,
    speedup: float,
    realized_work: np.ndarray,
    bucket_max_over_global_max: float,
) -> dict[str, Any]:
    realized = np.asarray(realized_work, dtype=float)
    mean_depth = float(np.mean(realized))
    std_depth = float(np.std(realized))
    return {
        "benchmark": config.benchmark,
        "model": config.model,
        "seed": seed,
        "scheduler_mode": scheduler_mode,
        "scheduler_label": scheduler_label,
        "is_analysis_upper_bound": is_analysis_upper_bound,
        "bucket_size": bucket_size,
        "speedup": speedup,
        "realized_depth_mean": mean_depth,
        "realized_depth_std": std_depth,
        "realized_depth_cv": std_depth / mean_depth if mean_depth else 0.0,
        "realized_depth_gini": _mean_step_gini(realized),
        "realized_depth_range": float(np.mean(np.ptp(realized, axis=1))),
        "bucket_max_over_global_max": bucket_max_over_global_max,
    }


def _mean_step_gini(values: np.ndarray) -> float:
    if values.ndim == 1:
        values = values[None, :]
    return float(np.mean([_gini(row) for row in values]))


def _gini(values: np.ndarray) -> float:
    sorted_values = np.sort(np.asarray(values, dtype=float))
    total = float(np.sum(sorted_values))
    if total == 0.0:
        return 0.0
    n = sorted_values.shape[0]
    weights = 2 * np.arange(1, n + 1) - n - 1
    return float(np.sum(weights * sorted_values) / (n * total))


def _scheduler_label(scheduler_mode: str) -> str:
    if scheduler_mode == "oracle_current":
        return "oracle_current (analysis-only upper bound)"
    if scheduler_mode == "oracle_previous":
        return "oracle_previous"
    return scheduler_mode


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {out_dir}. "
            "Pass --overwrite to replace oracle-gap outputs intentionally."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _csv_from_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _config_to_json(config: OracleGapConfig, *, profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "benchmark": config.benchmark,
        "model": config.model,
        "backend": config.backend,
        "dtype": config.dtype,
        "seeds": list(config.seeds),
        "num_chains": config.num_chains,
        "dimension": config.dimension,
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "bucket_sizes": list(config.bucket_sizes),
        "scheduler_modes": list(config.scheduler_modes),
        "predictor_beta": config.predictor_beta,
    }


def _as_str_tuple(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a string or list of strings")
    return tuple(str(item) for item in value)


def _as_int_tuple(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if not isinstance(value, Iterable) or isinstance(value, str):
        raise ValueError(f"{name} must be an integer or list of integers")
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only integers") from exc


def _as_positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed!r}")
    return parsed


def _as_positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {parsed!r}")
    return parsed


def _as_probability(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {parsed!r}")
    return parsed


def main(argv: list[str] | None = None) -> int:
    """Run the oracle-gap command."""
    args = parse_args(argv)
    backend = "unknown"
    try:
        raw_config = load_yaml_config(args.config)
        config = resolve_config(raw_config, profile=args.profile, backend_override=args.backend)
        backend = config.backend
        jax.config.update("jax_platform_name", config.backend)
        outputs = run_oracle_gap(config)
        write_outputs(
            args.out,
            config_path=args.config,
            profile=args.profile,
            config=config,
            outputs=outputs,
            overwrite=args.overwrite,
        )
    except (ConfigError, OSError, ValueError) as exc:
        print(
            "Oracle-gap run failed "
            f"(config={getattr(args, 'config', 'unknown')}, "
            f"profile={getattr(args, 'profile', 'unknown')}, "
            f"backend={backend}, "
            f"out={getattr(args, 'out', 'unknown')}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote oracle-gap summary: {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
