"""Benchmark runner for small scaling and ablation studies."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Callable, Iterable
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
from abnuts.nuts.bucketed import BucketedRunResult, run_bucketed
from abnuts.nuts.hmc import FixedHmcRunResult, run_fixed_hmc
from abnuts.nuts.independent import (
    IndependentChainRunResult,
    run_independent_chains_local,
)
from abnuts.nuts.monolithic import MonolithicRunResult, run_monolithic, run_monolithic_jit
from abnuts.nuts.predictors import PREDICTOR_MODES
from abnuts.profiling import TimingBreakdown

METHOD_ALIASES = {
    "independent": "independent_local",
    "independent_chain": "independent_local",
    "independent_chain_local": "independent_local",
}
METHODS = {"monolithic", "bucketed", "fixed_hmc", "independent_local"}


# Warm wall time is summarized by the median; every blocked repeat is retained.
# One shot is not reproducible on a shared node; see _repeated_blocked_call.
DEFAULT_WARM_REPEATS = 3


class BenchmarkConfig(NamedTuple):
    """Resolved benchmark configuration for one profile."""

    benchmark: str
    model: str
    methods: tuple[str, ...]
    backend: str
    dtype: str
    seeds: tuple[int, ...]
    num_chains: int
    chain_counts: tuple[int, ...]
    dimension: int
    dimensions: tuple[int, ...]
    model_config: dict[str, Any]
    num_steps: int
    step_size: float
    max_tree_depth: int
    hmc_num_leapfrog_steps: int
    bucket_sizes: tuple[int, ...]
    predictors: tuple[str, ...]
    predictor_beta: float


class TimedResult(NamedTuple):
    """A benchmark result plus the complete blocked warm-time sample."""

    result: Any
    elapsed_seconds: float
    timing: TimingBreakdown
    repeat_seconds: tuple[float, ...]
    cold_prime_seconds: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the benchmark runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", help="Override the configured benchmark model.")
    parser.add_argument("--method", help="Comma-separated method override.")
    parser.add_argument("--backend", choices=["cpu", "gpu"], help="Override configured backend.")
    parser.add_argument("--profile", default="tiny")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--predictors", help="Comma-separated predictor override.")
    parser.add_argument("--bucket-sizes", help="Comma-separated bucket-size override.")
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=DEFAULT_WARM_REPEATS,
        help="Blocked warm repeats per row; all are retained and the median is reported.",
    )
    parser.add_argument(
        "--enable-timing-breakdown",
        action="store_true",
        help="Run an extra cold/warm pass and record runtime component fields.",
    )
    parser.add_argument(
        "--enable-profiler-markers",
        action="store_true",
        help="Emit optional JAX profiler markers around profiled timing scopes.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_config(
    raw_config: dict[str, Any],
    *,
    profile: str,
    model_override: str | None,
    method_override: str | None,
    backend_override: str | None,
    predictor_override: str | None,
    bucket_size_override: str | None,
) -> BenchmarkConfig:
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
    if model_override is not None:
        merged["model"] = model_override
    if backend_override is not None:
        merged["backend"] = backend_override

    methods = _canonical_methods(
        _parse_str_list(method_override, name="--method")
        if method_override is not None
        else _as_str_tuple(merged.get("methods", ("bucketed",)), "methods")
    )
    unknown_methods = sorted(set(methods) - METHODS)
    if unknown_methods:
        valid = ", ".join(sorted(METHODS | set(METHOD_ALIASES)))
        raise ValueError(f"unsupported methods {unknown_methods!r}; expected one of {valid}")

    predictors = (
        _parse_str_list(predictor_override, name="--predictors")
        if predictor_override is not None
        else _as_str_tuple(merged.get("predictors", ("none", "random", "history")), "predictors")
    )
    unknown_predictors = sorted(set(predictors) - PREDICTOR_MODES)
    if unknown_predictors:
        valid = ", ".join(sorted(PREDICTOR_MODES))
        raise ValueError(f"unsupported predictors {unknown_predictors!r}; expected one of {valid}")

    bucket_sizes = (
        _parse_int_list(bucket_size_override, name="--bucket-sizes")
        if bucket_size_override is not None
        else _as_int_tuple(merged.get("bucket_sizes", (2, 4)), "bucket_sizes")
    )

    max_tree_depth = _as_positive_int(merged.get("max_tree_depth", 2), "max_tree_depth")
    hmc_num_leapfrog_steps = _as_positive_int(
        merged.get("hmc_num_leapfrog_steps", (1 << max_tree_depth) - 1),
        "hmc_num_leapfrog_steps",
    )
    config = BenchmarkConfig(
        benchmark=str(merged.get("benchmark", "funnel_ablation")),
        model=str(merged.get("model", "funnel")),
        methods=methods,
        backend=str(merged.get("backend", "cpu")),
        dtype=str(merged.get("dtype", "float32")),
        seeds=_as_int_tuple(merged.get("seeds", (merged.get("seed", 0),)), "seeds"),
        num_chains=_as_positive_int(merged.get("num_chains", 4), "num_chains"),
        chain_counts=_as_int_tuple(
            merged.get("chain_counts", (merged.get("num_chains", 4),)), "chain_counts"
        ),
        dimension=_as_positive_int(merged.get("dimension", 4), "dimension"),
        dimensions=_as_int_tuple(
            merged.get("dimensions", (merged.get("dimension", 4),)), "dimensions"
        ),
        model_config=_as_model_config(merged.get("model_config", {})),
        num_steps=_as_positive_int(merged.get("num_steps", 1), "num_steps"),
        step_size=_as_positive_float(merged.get("step_size", 0.03), "step_size"),
        max_tree_depth=max_tree_depth,
        hmc_num_leapfrog_steps=hmc_num_leapfrog_steps,
        bucket_sizes=bucket_sizes,
        predictors=predictors,
        predictor_beta=_as_probability(merged.get("predictor_beta", 0.9), "predictor_beta"),
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    """Validate resolved benchmark settings."""
    if config.backend not in {"cpu", "gpu"}:
        raise ValueError(f"run_benchmark supports backend='cpu' or 'gpu', got {config.backend!r}")
    if config.dtype != "float32":
        raise ValueError(f"run_benchmark currently supports dtype='float32', got {config.dtype!r}")
    if any(bucket_size <= 0 for bucket_size in config.bucket_sizes):
        raise ValueError(f"bucket_sizes must all be positive, got {config.bucket_sizes!r}")
    if not config.seeds:
        raise ValueError("seeds must contain at least one seed")
    if not config.predictors:
        raise ValueError("predictors must contain at least one mode")
    if not config.methods:
        raise ValueError("methods must contain at least one method")
    if any(num_chains <= 0 for num_chains in config.chain_counts):
        raise ValueError(f"chain_counts must all be positive, got {config.chain_counts!r}")
    if any(dimension <= 0 for dimension in config.dimensions):
        raise ValueError(f"dimensions must all be positive, got {config.dimensions!r}")


def run_ablation(
    config: BenchmarkConfig,
    *,
    enable_timing_breakdown: bool = False,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
    enable_profiler_markers: bool = False,
) -> list[dict[str, Any]]:
    """Run the configured funnel ablation and return summary rows."""
    rows: list[dict[str, Any]] = []

    for dimension in config.dimensions:
        model = get_model(
            config.model,
            dimension=dimension,
            model_config=config.model_config,
        )
        for num_chains in config.chain_counts:
            for seed in config.seeds:
                rows.extend(
                    _run_one_setting(
                        model,
                        config,
                        seed=seed,
                        num_chains=num_chains,
                        enable_timing_breakdown=enable_timing_breakdown,
                        warm_repeats=warm_repeats,
                        enable_profiler_markers=enable_profiler_markers,
                    )
                )
    return rows


def _run_one_setting(
    model: BenchmarkModel,
    config: BenchmarkConfig,
    *,
    seed: int,
    num_chains: int,
    enable_timing_breakdown: bool,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
    enable_profiler_markers: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model_metadata = model.metadata.as_dict()
    initial_positions = jnp.asarray(
        model.initial_position(
            key=seed,
            num_chains=num_chains,
            config=config.model_config,
        ),
        dtype=jnp.float32,
    )
    block_until_ready_tree(initial_positions)
    mono_timed = _time_monolithic(
        model,
        initial_positions,
        config,
        seed=seed,
        enable_timing_breakdown=enable_timing_breakdown,
        warm_repeats=warm_repeats,
    )
    mono_summary = _summarize_monolithic(mono_timed.result)
    mono_method_summary = _summarize_method(mono_timed.result)

    if "monolithic" in config.methods:
        rows.append(
            _make_summary_row(
                model=model,
                model_name=config.model,
                model_metadata=model_metadata,
                config=config,
                seed=seed,
                num_chains=num_chains,
                method="monolithic",
                baseline_type="nuts_reference",
                predictor="monolithic",
                bucket_size=0,
                timed=mono_timed,
                mono_timed=mono_timed,
                mono_summary=mono_summary,
                method_summary=mono_method_summary,
            )
        )

    if "bucketed" in config.methods:
        for predictor in config.predictors:
            for bucket_size in config.bucket_sizes:
                bucket_timed = _time_bucketed(
                    model,
                    initial_positions,
                    config,
                    seed=seed,
                    predictor=predictor,
                    bucket_size=bucket_size,
                    enable_timing_breakdown=enable_timing_breakdown,
                    warm_repeats=warm_repeats,
                    enable_profiler_markers=enable_profiler_markers,
                )
                bucket_summary = _summarize_bucketed(bucket_timed.result)
                rows.append(
                    _make_summary_row(
                        model=model,
                        model_name=config.model,
                        model_metadata=model_metadata,
                        config=config,
                        seed=seed,
                        num_chains=num_chains,
                        method="bucketed",
                        baseline_type="bucketed_nuts",
                        predictor=predictor,
                        bucket_size=bucket_size,
                        timed=bucket_timed,
                        mono_timed=mono_timed,
                        mono_summary=mono_summary,
                        method_summary=_summarize_method(bucket_timed.result),
                        bucket_summary=bucket_summary,
                    )
                )

    if "fixed_hmc" in config.methods:
        hmc_timed = _time_fixed_hmc(
            model,
            initial_positions,
            config,
            seed=seed,
            enable_timing_breakdown=enable_timing_breakdown,
        warm_repeats=warm_repeats,
        )
        rows.append(
            _make_summary_row(
                model=model,
                model_name=config.model,
                model_metadata=model_metadata,
                config=config,
                seed=seed,
                num_chains=num_chains,
                method="fixed_hmc",
                baseline_type="fixed_length_hmc",
                predictor="fixed_hmc",
                bucket_size=0,
                timed=hmc_timed,
                mono_timed=mono_timed,
                mono_summary=mono_summary,
                method_summary=_summarize_method(hmc_timed.result),
                hmc_num_leapfrog_steps=hmc_timed.result.num_leapfrog_steps,
            )
        )

    if "independent_local" in config.methods:
        independent_timed = _time_independent_local(
            model,
            initial_positions,
            config,
            seed=seed,
            enable_timing_breakdown=enable_timing_breakdown,
        warm_repeats=warm_repeats,
        )
        rows.append(
            _make_summary_row(
                model=model,
                model_name=config.model,
                model_metadata=model_metadata,
                config=config,
                seed=seed,
                num_chains=num_chains,
                method="independent_local",
                baseline_type=independent_timed.result.baseline_type,
                predictor="independent_local",
                bucket_size=0,
                timed=independent_timed,
                mono_timed=mono_timed,
                mono_summary=mono_summary,
                method_summary=_summarize_method(independent_timed.result),
                independent_baseline_kind=independent_timed.result.baseline_type,
            )
        )
    return rows


def _time_monolithic(
    model: BenchmarkModel,
    initial_positions: Any,
    config: BenchmarkConfig,
    *,
    seed: int,
    enable_timing_breakdown: bool,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
) -> TimedResult:
    def run_once() -> MonolithicRunResult:
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
        enable_timing_breakdown=enable_timing_breakdown,
        warm_repeats=warm_repeats,
        timing_from_result=None,
    )


def _time_bucketed(
    model: BenchmarkModel,
    initial_positions: Any,
    config: BenchmarkConfig,
    *,
    seed: int,
    predictor: str,
    bucket_size: int,
    enable_timing_breakdown: bool,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
    enable_profiler_markers: bool,
) -> TimedResult:
    def run_once() -> BucketedRunResult:
        return run_bucketed(
            model,
            initial_positions,
            jr.PRNGKey(seed),
            num_steps=config.num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            bucket_size=bucket_size,
            predictor=predictor,
            predictor_beta=config.predictor_beta,
            dtype=jnp.float32,
        )

    def run_profiled_once() -> BucketedRunResult:
        return run_bucketed(
            model,
            initial_positions,
            jr.PRNGKey(seed),
            num_steps=config.num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            bucket_size=bucket_size,
            predictor=predictor,
            predictor_beta=config.predictor_beta,
            dtype=jnp.float32,
            enable_timing_breakdown=True,
            enable_profiler_markers=enable_profiler_markers,
        )

    timing_runner = run_profiled_once if enable_timing_breakdown else run_once
    return _time_runner(
        timing_runner,
        warm_runner=run_profiled_once if enable_timing_breakdown else None,
        enable_timing_breakdown=enable_timing_breakdown,
        warm_repeats=warm_repeats,
        timing_from_result=lambda result: result.timing,
    )


def _time_fixed_hmc(
    model: BenchmarkModel,
    initial_positions: Any,
    config: BenchmarkConfig,
    *,
    seed: int,
    enable_timing_breakdown: bool,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
) -> TimedResult:
    def run_once() -> FixedHmcRunResult:
        return run_fixed_hmc(
            model,
            initial_positions,
            jr.PRNGKey(seed),
            num_steps=config.num_steps,
            step_size=config.step_size,
            num_leapfrog_steps=config.hmc_num_leapfrog_steps,
            dtype=jnp.float32,
        )

    return _time_runner(
        run_once,
        enable_timing_breakdown=enable_timing_breakdown,
        warm_repeats=warm_repeats,
        timing_from_result=None,
    )


def _time_independent_local(
    model: BenchmarkModel,
    initial_positions: Any,
    config: BenchmarkConfig,
    *,
    seed: int,
    enable_timing_breakdown: bool,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
) -> TimedResult:
    def run_once() -> IndependentChainRunResult:
        return run_independent_chains_local(
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
        enable_timing_breakdown=enable_timing_breakdown,
        warm_repeats=warm_repeats,
        timing_from_result=None,
    )


def _time_runner(
    runner: Callable[[], Any],
    *,
    enable_timing_breakdown: bool,
    timing_from_result: Callable[[Any], TimingBreakdown] | None,
    warm_runner: Callable[[], Any] | None = None,
    warm_repeats: int = DEFAULT_WARM_REPEATS,
) -> TimedResult:
    if not enable_timing_breakdown:
        _cold_result, cold_prime_seconds = _elapsed_blocked_call(runner)
        result, elapsed_seconds, repeat_seconds, _ = _repeated_blocked_call(
            runner,
            warm_repeats,
        )
        return TimedResult(
            result=result,
            elapsed_seconds=elapsed_seconds,
            timing=TimingBreakdown(),
            repeat_seconds=repeat_seconds,
            cold_prime_seconds=cold_prime_seconds,
        )

    _cold_result, cold_run_seconds = _elapsed_blocked_call(runner)
    warm_result, warm_iteration_seconds, repeat_seconds, representative_timing = (
        _repeated_blocked_call(
            warm_runner or runner,
            warm_repeats,
            timing_from_result=timing_from_result,
        )
    )
    if timing_from_result is None:
        timing = TimingBreakdown(
            enabled=True,
            executor_seconds=warm_iteration_seconds,
            total_profiled_seconds=warm_iteration_seconds,
        )
    else:
        assert representative_timing is not None
        timing = representative_timing
    timing = timing.with_outer_timings(
        cold_run_seconds=cold_run_seconds,
        warm_iteration_seconds=warm_iteration_seconds,
    )
    return TimedResult(
        result=warm_result,
        elapsed_seconds=warm_iteration_seconds,
        timing=timing,
        repeat_seconds=repeat_seconds,
        cold_prime_seconds=cold_run_seconds,
    )


def _repeated_blocked_call(
    runner: Callable[[], Any],
    repeats: int,
    *,
    timing_from_result: Callable[[Any], TimingBreakdown] | None = None,
) -> tuple[Any, float, tuple[float, ...], TimingBreakdown | None]:
    """Run blocked warm calls and retain the full sample and median repeat.

    Results are deterministic under fixed keys, so only the final sampler result
    is retained.  Component metadata are small and are retained for every
    repeat; the component record returned is from the observed repeat nearest
    the sample median, ensuring that outer and component timings describe the
    same execution.
    """
    if repeats <= 0:
        raise ValueError(f"warm_repeats must be positive, got {repeats!r}")
    result = None
    elapsed_values: list[float] = []
    timing_values: list[TimingBreakdown | None] = []
    for _ in range(repeats):
        result, elapsed_seconds = _elapsed_blocked_call(runner)
        elapsed_values.append(elapsed_seconds)
        timing_values.append(
            timing_from_result(result) if timing_from_result is not None else None
        )
    median_seconds = float(np.median(np.asarray(elapsed_values, dtype=float)))
    representative_index = min(
        range(repeats),
        key=lambda index: abs(elapsed_values[index] - median_seconds),
    )
    return (
        result,
        median_seconds,
        tuple(elapsed_values),
        timing_values[representative_index],
    )


def _elapsed_blocked_call(runner: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = runner()
    result = block_until_ready_tree(result)
    stop = time.perf_counter()
    return result, stop - start


def _summarize_monolithic(result: MonolithicRunResult) -> dict[str, Any]:
    info = result.transition_info
    return {
        "mono_total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
        "mono_divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
        "mono_max_realized_tree_depth": int(np.max(np.asarray(info.realized_tree_depth))),
    }


def _summarize_bucketed(result: BucketedRunResult) -> dict[str, Any]:
    info = result.transition_info
    final_plan = result.bucket_plans[-1]
    return {
        "bucket_total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
        "bucket_divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
        "bucket_max_realized_tree_depth": int(np.max(np.asarray(info.realized_tree_depth))),
        "bucket_num_buckets": int(final_plan.num_buckets),
        "bucket_padding_count": int(final_plan.padding_count),
        "bucket_padding_ratio": float(final_plan.padding_ratio),
        "bucket_hvp_overhead_seconds": float(result.hvp_overhead_seconds),
        "bucket_hvp_call_count": int(result.hvp_call_count),
    }


def _summarize_method(
    result: MonolithicRunResult
    | BucketedRunResult
    | FixedHmcRunResult
    | IndependentChainRunResult,
) -> dict[str, Any]:
    info = result.transition_info
    return {
        "method_total_leapfrog_count": int(np.sum(np.asarray(info.leapfrog_count))),
        "method_divergence_count": int(np.sum(np.asarray(info.divergence_flag))),
        "method_max_realized_tree_depth": int(
            np.max(np.asarray(info.realized_tree_depth))
        ),
    }


def _make_summary_row(
    *,
    model: BenchmarkModel,
    model_name: str,
    model_metadata: dict[str, Any],
    config: BenchmarkConfig,
    seed: int,
    num_chains: int,
    method: str,
    baseline_type: str,
    predictor: str,
    bucket_size: int,
    timed: TimedResult,
    mono_timed: TimedResult,
    mono_summary: dict[str, Any],
    method_summary: dict[str, Any],
    bucket_summary: dict[str, Any] | None = None,
    hmc_num_leapfrog_steps: int | str = "",
    independent_baseline_kind: str = "",
) -> dict[str, Any]:
    speedup = mono_timed.elapsed_seconds / timed.elapsed_seconds
    row: dict[str, Any] = {
        "model": model_name,
        "model_family": model_metadata.get("model_family", ""),
        "parameterization": model_metadata.get("parameterization", ""),
        "seed": seed,
        "method": method,
        "baseline_type": baseline_type,
        "predictor": predictor,
        "bucket_size": bucket_size,
        "num_chains": num_chains,
        "dimension": model.dimension,
        "num_groups": model_metadata.get("num_groups", ""),
        "observations_per_group": model_metadata.get("observations_per_group", ""),
        "num_observations": model_metadata.get("num_observations", ""),
        "num_features": model_metadata.get("num_features", ""),
        "time_length": model_metadata.get("time_length", ""),
        "kernel": model_metadata.get("kernel", ""),
        "input_dimension": model_metadata.get("input_dimension", ""),
        "data_seed": model_metadata.get("data_seed", ""),
        "jitter": model_metadata.get("jitter", ""),
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "hmc_num_leapfrog_steps": hmc_num_leapfrog_steps,
        "independent_baseline_kind": independent_baseline_kind,
        "predictor_beta": config.predictor_beta,
        "time_seconds": timed.elapsed_seconds,
        "speedup_vs_monolithic": speedup,
        "t_mono": mono_timed.elapsed_seconds,
        "t_bucket": timed.elapsed_seconds,
        "speedup": speedup,
        "timing_cold_prime_seconds": timed.cold_prime_seconds,
        "timing_warm_repeat_count": len(timed.repeat_seconds),
        "timing_warm_min_seconds": min(timed.repeat_seconds),
        "timing_warm_q1_seconds": float(np.percentile(timed.repeat_seconds, 25)),
        "timing_warm_median_seconds": float(np.median(timed.repeat_seconds)),
        "timing_warm_q3_seconds": float(np.percentile(timed.repeat_seconds, 75)),
        "timing_warm_max_seconds": max(timed.repeat_seconds),
        "timing_warm_repeat_seconds_json": json.dumps(timed.repeat_seconds),
        "mono_warm_repeat_seconds_json": json.dumps(mono_timed.repeat_seconds),
        **mono_summary,
        **method_summary,
        "bucket_total_leapfrog_count": method_summary["method_total_leapfrog_count"],
        "bucket_divergence_count": method_summary["method_divergence_count"],
        "bucket_max_realized_tree_depth": method_summary[
            "method_max_realized_tree_depth"
        ],
        "bucket_num_buckets": "",
        "bucket_padding_count": 0,
        "bucket_padding_ratio": 0.0,
        "bucket_hvp_overhead_seconds": 0.0,
        "bucket_hvp_call_count": 0,
    }
    if bucket_summary is not None:
        row.update(bucket_summary)
    row.update(timed.timing.as_summary_fields(elapsed_seconds=timed.elapsed_seconds))
    return row


def write_outputs(
    out_dir: Path,
    *,
    config_path: Path,
    profile: str,
    config: BenchmarkConfig,
    rows: list[dict[str, Any]],
    overwrite: bool,
    enable_timing_breakdown: bool,
    warm_repeats: int,
    enable_profiler_markers: bool,
) -> None:
    """Write benchmark manifest and per-config summary CSV."""
    _prepare_output_dir(out_dir, overwrite=overwrite)
    summary_csv = _csv_from_rows(rows)
    timing_repetitions_csv = _csv_from_rows(_timing_repetition_rows(rows))
    resolved_config = _config_to_json(config, profile=profile)
    model_metadata = _model_metadata_for_config(config)
    manifest = build_result_manifest(
        command="python -m abnuts.experiments.run_benchmark",
        config_path=config_path,
        output_dir=out_dir,
        config=resolved_config,
        extra={
            "profile": profile,
            "benchmark": config.benchmark,
            "timing_breakdown_enabled": enable_timing_breakdown,
            "warm_repeats": warm_repeats,
            "warm_timing_summary": "median of warm_repeats blocked repeats",
            "warm_timing_dispersion": "IQR with every repeat retained",
            "profiler_markers_enabled": enable_profiler_markers,
            "model_metadata": model_metadata,
            "row_count": len(rows),
            "outputs": {
                "summary_csv": "summary.csv",
                "timing_repetitions_csv": "timing_repetitions.csv",
            },
        },
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.csv").write_text(summary_csv, encoding="utf-8")
    (out_dir / "timing_repetitions.csv").write_text(
        timing_repetitions_csv,
        encoding="utf-8",
    )


def _timing_repetition_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand summary-embedded timing samples into tidy raw rows."""
    repetition_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows):
        method_times = json.loads(str(row["timing_warm_repeat_seconds_json"]))
        mono_times = json.loads(str(row["mono_warm_repeat_seconds_json"]))
        for repeat_index, method_seconds in enumerate(method_times):
            repetition_rows.append(
                {
                    "row_number": row_number,
                    "model": row["model"],
                    "seed": row["seed"],
                    "method": row["method"],
                    "predictor": row["predictor"],
                    "bucket_size": row["bucket_size"],
                    "num_chains": row["num_chains"],
                    "dimension": row["dimension"],
                    "repeat": repeat_index,
                    "method_seconds": method_seconds,
                    "monolithic_seconds_same_index": mono_times[repeat_index],
                    "pairing_note": (
                        "repeat indices align samples but methods were timed sequentially; "
                        "use independent-resample intervals"
                    ),
                }
            )
    return repetition_rows


def _prepare_output_dir(out_dir: Path, *, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {out_dir}. "
            "Pass --overwrite to replace benchmark outputs intentionally."
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


def _config_to_json(config: BenchmarkConfig, *, profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "benchmark": config.benchmark,
        "model": config.model,
        "methods": list(config.methods),
        "backend": config.backend,
        "dtype": config.dtype,
        "seeds": list(config.seeds),
        "num_chains": config.num_chains,
        "chain_counts": list(config.chain_counts),
        "dimension": config.dimension,
        "dimensions": list(config.dimensions),
        "model_config": config.model_config,
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "hmc_num_leapfrog_steps": config.hmc_num_leapfrog_steps,
        "bucket_sizes": list(config.bucket_sizes),
        "predictors": list(config.predictors),
        "predictor_beta": config.predictor_beta,
    }


def _model_metadata_for_config(config: BenchmarkConfig) -> dict[str, Any]:
    unique_dimensions = tuple(dict.fromkeys(config.dimensions))
    if len(unique_dimensions) == 1:
        model = get_model(
            config.model,
            dimension=unique_dimensions[0],
            model_config=config.model_config,
        )
        return model.metadata.as_dict()
    return {
        "model": config.model,
        "by_dimension": [
            get_model(
                config.model,
                dimension=dimension,
                model_config=config.model_config,
            ).metadata.as_dict()
            for dimension in unique_dimensions
        ],
    }


def _default_config_path(model_name: str | None) -> Path:
    if model_name is not None and model_name.startswith("eight_schools_"):
        return Path("configs/eight_schools.yaml")
    if model_name == "hierarchical_logistic":
        return Path("configs/hierarchical_logistic.yaml")
    if model_name == "stochastic_volatility":
        return Path("configs/stochastic_volatility.yaml")
    if model_name == "gaussian_process":
        return Path("configs/gaussian_process.yaml")
    return Path("configs/funnel_ablation.yaml")


def _parse_str_list(raw: str, *, name: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def _canonical_methods(values: tuple[str, ...]) -> tuple[str, ...]:
    canonical: list[str] = []
    for value in values:
        method = METHOD_ALIASES.get(value, value)
        if method not in canonical:
            canonical.append(method)
    return tuple(canonical)


def _parse_int_list(raw: str, *, name: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


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


def _as_model_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("model_config must be a mapping")
    config: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"model_config keys must be strings, got {key!r}")
        config[key] = item
    return config


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark command."""
    args = parse_args(argv)
    backend = "unknown"
    config_path = args.config or _default_config_path(args.model)
    try:
        raw_config = load_yaml_config(config_path)
        config = resolve_config(
            raw_config,
            profile=args.profile,
            model_override=args.model,
            method_override=args.method,
            backend_override=args.backend,
            predictor_override=args.predictors,
            bucket_size_override=args.bucket_sizes,
        )
        backend = config.backend
        jax.config.update("jax_platform_name", config.backend)
        rows = run_ablation(
            config,
            enable_timing_breakdown=args.enable_timing_breakdown,
            warm_repeats=args.warm_repeats,
            enable_profiler_markers=args.enable_profiler_markers,
        )
        write_outputs(
            args.out,
            config_path=config_path,
            profile=args.profile,
            config=config,
            rows=rows,
            overwrite=args.overwrite,
            enable_timing_breakdown=args.enable_timing_breakdown,
            warm_repeats=args.warm_repeats,
            enable_profiler_markers=args.enable_profiler_markers,
        )
    except (ConfigError, OSError, ValueError) as exc:
        print(
            "Benchmark run failed "
            f"(config={config_path}, "
            f"profile={getattr(args, 'profile', 'unknown')}, "
            f"backend={backend}, "
            f"out={getattr(args, 'out', 'unknown')}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote benchmark summary: {args.out / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
