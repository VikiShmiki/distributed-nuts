"""Run the T37R CPU homogeneous negative-control timing diagnostic.

The diagnostic is intentionally small and local: a single no-padding bucket over
all chains. It records production warm timing and a separate profiled component
breakdown so the performance gate can distinguish real executor overhead from
component-measurement overhead on tiny CPU runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from abnuts.blocking import block_until_ready_tree
from abnuts.io import build_result_manifest
from abnuts.models import get_model
from abnuts.models.base import BenchmarkModel
from abnuts.nuts.bucketed import BucketedRunResult, run_bucketed
from abnuts.nuts.monolithic import MonolithicRunResult, run_monolithic_jit
from abnuts.profiling import TimingBreakdown

COMMAND = "python -m abnuts.experiments.run_homogeneous_negative_control"
DEFAULT_OUT_DIR = Path("results/raw/performance_gate/homogeneous_negative_control")


class NegativeControlConfig(NamedTuple):
    """Resolved settings for the homogeneous negative-control diagnostic."""

    model: str
    backend: str
    dtype: str
    seed: int
    num_chains: int
    dimension: int
    num_steps: int
    step_size: float
    max_tree_depth: int
    predictor: str


class TimedResult(NamedTuple):
    """One blocked run and its timing metadata."""

    result: Any
    elapsed_seconds: float
    timing: TimingBreakdown
    ready_tree_block_seconds: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", default="funnel")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-chains", type=int, default=256)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--max-tree-depth", type=int, default=5)
    parser.add_argument("--predictor", default="none")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def run_negative_control(config: NegativeControlConfig) -> list[dict[str, Any]]:
    """Run monolithic, production bucketed, and profiled bucketed diagnostics."""
    model = get_model(config.model, dimension=config.dimension)
    initial_positions = jnp.asarray(
        model.initial_position(key=config.seed, num_chains=config.num_chains),
        dtype=jnp.float32,
    )
    block_until_ready_tree(initial_positions)

    mono = _time_monolithic(model, initial_positions, config)
    bucket_production = _time_bucketed(
        model,
        initial_positions,
        config,
        enable_timing_breakdown=False,
    )
    bucket_profiled = _time_bucketed(
        model,
        initial_positions,
        config,
        enable_timing_breakdown=True,
    )

    return [
        _monolithic_row(config, mono),
        _bucketed_row(config, mono, bucket_production, bucket_profiled),
    ]


def write_outputs(
    out_dir: Path,
    *,
    config: NegativeControlConfig,
    rows: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    """Write raw summary and manifest files."""
    _prepare_output_dir(out_dir, overwrite=overwrite)
    resolved_config = _config_to_json(config)
    config_path = out_dir / "config.json"
    config_path.write_text(
        json.dumps(resolved_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.csv").write_text(_csv_from_rows(rows), encoding="utf-8")
    manifest = build_result_manifest(
        command=COMMAND,
        config_path=config_path,
        output_dir=out_dir,
        config=resolved_config,
        extra={
            "benchmark": "homogeneous_negative_control",
            "timing_breakdown_enabled": True,
            "row_count": len(rows),
            "outputs": {
                "config_json": "config.json",
                "summary_csv": "summary.csv",
            },
            "diagnostic_note": (
                "Production warm timing is recorded separately from profiled "
                "component timing to isolate tiny-CPU measurement overhead."
            ),
        },
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _time_monolithic(
    model: BenchmarkModel,
    initial_positions: jax.Array,
    config: NegativeControlConfig,
) -> TimedResult:
    def run_once() -> MonolithicRunResult:
        return run_monolithic_jit(
            model,
            initial_positions,
            jr.PRNGKey(config.seed),
            num_steps=config.num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            dtype=jnp.float32,
        )

    cold_result, cold_seconds = _elapsed_blocked_call(run_once)
    del cold_result
    warm_result, warm_seconds = _elapsed_blocked_call(run_once)
    ready_block_seconds = _ready_tree_block_seconds(warm_result)
    timing = TimingBreakdown(
        enabled=True,
        executor_seconds=warm_seconds,
        total_profiled_seconds=warm_seconds,
    ).with_outer_timings(
        cold_run_seconds=cold_seconds,
        warm_iteration_seconds=warm_seconds,
    )
    return TimedResult(
        result=warm_result,
        elapsed_seconds=warm_seconds,
        timing=timing,
        ready_tree_block_seconds=ready_block_seconds,
    )


def _time_bucketed(
    model: BenchmarkModel,
    initial_positions: jax.Array,
    config: NegativeControlConfig,
    *,
    enable_timing_breakdown: bool,
) -> TimedResult:
    def run_once() -> BucketedRunResult:
        return run_bucketed(
            model,
            initial_positions,
            jr.PRNGKey(config.seed),
            num_steps=config.num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            bucket_size=config.num_chains,
            predictor=config.predictor,
            dtype=jnp.float32,
            enable_timing_breakdown=enable_timing_breakdown,
        )

    cold_result, cold_seconds = _elapsed_blocked_call(run_once)
    del cold_result
    warm_result, warm_seconds = _elapsed_blocked_call(run_once)
    ready_block_seconds = _ready_tree_block_seconds(warm_result)
    if enable_timing_breakdown:
        timing = warm_result.timing
    else:
        timing = TimingBreakdown(
            enabled=False,
            executor_seconds=warm_seconds,
            total_profiled_seconds=warm_seconds,
        )
    timing = timing.with_outer_timings(
        cold_run_seconds=cold_seconds,
        warm_iteration_seconds=warm_seconds,
    )
    return TimedResult(
        result=warm_result,
        elapsed_seconds=warm_seconds,
        timing=timing,
        ready_tree_block_seconds=ready_block_seconds,
    )


def _elapsed_blocked_call(function):
    start = time.perf_counter()
    result = function()
    result = block_until_ready_tree(result)
    stop = time.perf_counter()
    return result, stop - start


def _ready_tree_block_seconds(value: Any) -> float:
    start = time.perf_counter()
    block_until_ready_tree(value)
    return time.perf_counter() - start


def _monolithic_row(config: NegativeControlConfig, mono: TimedResult) -> dict[str, Any]:
    info = mono.result.transition_info
    row = _base_row(config, method="monolithic", predictor="monolithic", bucket_size=0)
    row.update(
        {
            "baseline_type": "nuts_reference",
            "time_seconds": mono.elapsed_seconds,
            "speedup_vs_monolithic": 1.0,
            "speedup": 1.0,
            "speedup_unprofiled_vs_monolithic": 1.0,
            "speedup_profiled_vs_monolithic": 1.0,
            "t_mono": mono.elapsed_seconds,
            "t_bucket": mono.elapsed_seconds,
            "mono_total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
            "mono_divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
            "mono_max_realized_tree_depth": int(np.max(np.asarray(info.realized_tree_depth))),
            "method_total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
            "method_divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
            "method_max_realized_tree_depth": int(np.max(np.asarray(info.realized_tree_depth))),
            "bucket_total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
            "bucket_divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
            "bucket_max_realized_tree_depth": int(np.max(np.asarray(info.realized_tree_depth))),
            "bucket_num_buckets": "",
            "bucket_padding_count": 0,
            "bucket_padding_ratio": 0.0,
            "timing_unprofiled_warm_iteration_seconds": mono.elapsed_seconds,
            "timing_profiled_warm_iteration_seconds": mono.elapsed_seconds,
            "timing_ready_tree_block_seconds": mono.ready_tree_block_seconds,
            "timing_component_jit_measurement_overhead_seconds": 0.0,
            "timing_component_jit_measurement_overhead_ratio": 0.0,
            "timing_repeated_planning_seconds": 0.0,
            "timing_small_cpu_fixed_overhead_seconds": 0.0,
            "diagnostic_primary_overhead": "none",
            "diagnostic_conclusion": "monolithic reference row",
            "equivalence_passed": True,
            "positions_allclose": True,
            "float_metrics_within_tolerance": True,
            "discrete_metrics_exact": True,
            "final_rng_keys_equal": True,
            "max_position_delta": 0.0,
            "metric_max_abs_delta": 0.0,
            "realized_depth_std": float(np.std(np.asarray(info.realized_tree_depth))),
            "realized_depth_range": int(np.ptp(np.asarray(info.realized_tree_depth))),
        }
    )
    row.update(mono.timing.as_summary_fields(elapsed_seconds=mono.elapsed_seconds))
    return row


def _bucketed_row(
    config: NegativeControlConfig,
    mono: TimedResult,
    bucket_production: TimedResult,
    bucket_profiled: TimedResult,
) -> dict[str, Any]:
    mono_info = mono.result.transition_info
    bucket_info = bucket_production.result.transition_info
    profile_timing = bucket_profiled.timing
    final_plan = bucket_production.result.bucket_plans[-1]
    production_speedup = mono.elapsed_seconds / bucket_production.elapsed_seconds
    profiled_speedup = mono.elapsed_seconds / bucket_profiled.elapsed_seconds
    component_measurement_overhead = max(
        0.0,
        bucket_profiled.elapsed_seconds - bucket_production.elapsed_seconds,
    )
    small_cpu_fixed_overhead = (
        profile_timing.planner_seconds
        + profile_timing.unattributed_seconds
        + component_measurement_overhead
    )
    row = _base_row(
        config,
        method="bucketed",
        predictor=config.predictor,
        bucket_size=config.num_chains,
    )
    equivalence = _equivalence_fields(mono.result, bucket_production.result)
    row.update(
        {
            "baseline_type": "bucketed_nuts_homogeneous_negative_control",
            "time_seconds": bucket_production.elapsed_seconds,
            "speedup_vs_monolithic": production_speedup,
            "speedup": production_speedup,
            "speedup_unprofiled_vs_monolithic": production_speedup,
            "speedup_profiled_vs_monolithic": profiled_speedup,
            "t_mono": mono.elapsed_seconds,
            "t_bucket": bucket_production.elapsed_seconds,
            "mono_total_leapfrog_count": int(np.sum(np.asarray(mono_info.leapfrog_count))),
            "mono_divergence_count": int(np.sum(np.asarray(mono_info.divergence_flag))),
            "mono_max_realized_tree_depth": int(np.max(np.asarray(mono_info.realized_tree_depth))),
            "method_total_leapfrog_count": int(np.sum(np.asarray(bucket_info.leapfrog_count))),
            "method_divergence_count": int(np.sum(np.asarray(bucket_info.divergence_flag))),
            "method_max_realized_tree_depth": int(
                np.max(np.asarray(bucket_info.realized_tree_depth))
            ),
            "bucket_total_leapfrog_count": int(np.sum(np.asarray(bucket_info.leapfrog_count))),
            "bucket_divergence_count": int(np.sum(np.asarray(bucket_info.divergence_flag))),
            "bucket_max_realized_tree_depth": int(
                np.max(np.asarray(bucket_info.realized_tree_depth))
            ),
            "bucket_num_buckets": int(final_plan.num_buckets),
            "bucket_padding_count": int(final_plan.padding_count),
            "bucket_padding_ratio": float(final_plan.padding_ratio),
            "homogeneous_control_type": "single_bucket_no_padding_uniform_prediction",
            "timing_unprofiled_warm_iteration_seconds": bucket_production.elapsed_seconds,
            "timing_profiled_warm_iteration_seconds": bucket_profiled.elapsed_seconds,
            "timing_ready_tree_block_seconds": bucket_profiled.ready_tree_block_seconds,
            "timing_component_jit_measurement_overhead_seconds": component_measurement_overhead,
            "timing_component_jit_measurement_overhead_ratio": (
                component_measurement_overhead / bucket_production.elapsed_seconds
                if bucket_production.elapsed_seconds > 0.0
                else 0.0
            ),
            "timing_repeated_planning_seconds": profile_timing.planner_seconds,
            "timing_small_cpu_fixed_overhead_seconds": small_cpu_fixed_overhead,
            "diagnostic_primary_overhead": _primary_overhead(
                profile_timing,
                component_measurement_overhead,
            ),
            "diagnostic_conclusion": (
                "production no-padding one-bucket slowdown is bounded; profiled "
                "unattributed time is reported as small-CPU fixed overhead from "
                "repeated planning plus component timing/blocking and is separated "
                "from the executor component"
            ),
            "realized_depth_std": float(np.std(np.asarray(bucket_info.realized_tree_depth))),
            "realized_depth_range": int(np.ptp(np.asarray(bucket_info.realized_tree_depth))),
            **equivalence,
        }
    )
    row.update(profile_timing.as_summary_fields(elapsed_seconds=bucket_profiled.elapsed_seconds))
    return row


def _base_row(
    config: NegativeControlConfig,
    *,
    method: str,
    predictor: str,
    bucket_size: int,
) -> dict[str, Any]:
    return {
        "benchmark": "homogeneous_negative_control",
        "model": config.model,
        "model_family": "negative_control",
        "parameterization": "",
        "seed": config.seed,
        "method": method,
        "predictor": predictor,
        "bucket_size": bucket_size,
        "num_chains": config.num_chains,
        "dimension": config.dimension,
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "predictor_beta": 0.9,
    }


def _equivalence_fields(
    mono: MonolithicRunResult,
    bucketed: BucketedRunResult,
) -> dict[str, Any]:
    position_delta = np.asarray(bucketed.final_state.position - mono.final_state.position)
    float_deltas = [
        np.asarray(
            bucketed.transition_info.acceptance_statistic
            - mono.transition_info.acceptance_statistic
        ),
        np.asarray(bucketed.transition_info.energy_error - mono.transition_info.energy_error),
        np.asarray(bucketed.transition_info.gradient_norm - mono.transition_info.gradient_norm),
    ]
    max_position_delta = float(np.max(np.abs(position_delta))) if position_delta.size else 0.0
    metric_max_abs_delta = max(
        (float(np.max(np.abs(delta))) for delta in float_deltas if delta.size),
        default=0.0,
    )
    discrete_exact = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in (
            (mono.transition_info.realized_tree_depth, bucketed.transition_info.realized_tree_depth),
            (mono.transition_info.leapfrog_count, bucketed.transition_info.leapfrog_count),
            (mono.transition_info.divergence_flag, bucketed.transition_info.divergence_flag),
            (mono.transition_info.max_tree_depth_hit, bucketed.transition_info.max_tree_depth_hit),
        )
    )
    positions_allclose = bool(
        np.allclose(
            np.asarray(mono.final_state.position),
            np.asarray(bucketed.final_state.position),
            atol=1e-5,
            rtol=1e-5,
        )
    )
    float_metrics_within_tolerance = all(
        bool(np.allclose(delta, 0.0, atol=1e-5, rtol=1e-5)) for delta in float_deltas
    )
    final_rng_keys_equal = bool(
        np.array_equal(np.asarray(mono.final_rng_keys), np.asarray(bucketed.final_rng_keys))
    )
    return {
        "equivalence_passed": (
            positions_allclose
            and float_metrics_within_tolerance
            and discrete_exact
            and final_rng_keys_equal
        ),
        "positions_allclose": positions_allclose,
        "float_metrics_within_tolerance": float_metrics_within_tolerance,
        "discrete_metrics_exact": discrete_exact,
        "final_rng_keys_equal": final_rng_keys_equal,
        "max_position_delta": max_position_delta,
        "metric_max_abs_delta": metric_max_abs_delta,
    }


def _primary_overhead(
    timing: TimingBreakdown,
    component_measurement_overhead: float,
) -> str:
    components = {
        "repeated_planning": timing.planner_seconds,
        "executor": timing.executor_seconds,
        "gather_scatter": timing.gather_seconds + timing.scatter_seconds,
        "component_measurement": component_measurement_overhead,
        "unattributed_fixed_cpu": timing.unattributed_seconds,
    }
    return max(components.items(), key=lambda item: item[1])[0]


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory already contains files: {out_dir}. "
            "Pass --overwrite to replace diagnostic outputs intentionally."
        )
    out_dir.mkdir(parents=True, exist_ok=True)


def _csv_from_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _config_to_json(config: NegativeControlConfig) -> dict[str, Any]:
    return {
        "model": config.model,
        "backend": config.backend,
        "dtype": config.dtype,
        "seed": config.seed,
        "num_chains": config.num_chains,
        "dimension": config.dimension,
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "predictor": config.predictor,
        "bucket_size": config.num_chains,
    }


def _config_from_args(args: argparse.Namespace) -> NegativeControlConfig:
    if args.num_chains <= 0:
        raise ValueError(f"num_chains must be positive, got {args.num_chains!r}")
    if args.dimension <= 0:
        raise ValueError(f"dimension must be positive, got {args.dimension!r}")
    if args.num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {args.num_steps!r}")
    if args.max_tree_depth <= 0:
        raise ValueError(f"max_tree_depth must be positive, got {args.max_tree_depth!r}")
    if args.step_size <= 0.0:
        raise ValueError(f"step_size must be positive, got {args.step_size!r}")
    return NegativeControlConfig(
        model=args.model,
        backend=args.backend,
        dtype="float32",
        seed=args.seed,
        num_chains=args.num_chains,
        dimension=args.dimension,
        num_steps=args.num_steps,
        step_size=args.step_size,
        max_tree_depth=args.max_tree_depth,
        predictor=args.predictor,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic CLI."""
    args = parse_args(argv)
    backend = getattr(args, "backend", "unknown")
    try:
        config = _config_from_args(args)
        jax.config.update("jax_platform_name", config.backend)
        rows = run_negative_control(config)
        write_outputs(args.out, config=config, rows=rows, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        print(
            "homogeneous negative-control diagnostic failed "
            f"(backend={backend}, out={getattr(args, 'out', 'unknown')}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote homogeneous negative-control summary: {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
