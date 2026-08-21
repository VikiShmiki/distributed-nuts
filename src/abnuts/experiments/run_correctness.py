"""Tiny correctness runner for monolithic NUTS reference executions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from abnuts.analysis.diagnostics import (
    DiagnosticsDependencyError,
    diagnostics_rows_equal,
    diagnostics_rows_to_csv,
    summarize_trace_diagnostics,
)
from abnuts.blocking import block_until_ready_tree
from abnuts.io import stable_json_hash, write_manifest
from abnuts.models import available_models, get_model
from abnuts.nuts.bucketed import BucketedRunResult, run_bucketed
from abnuts.nuts.monolithic import MonolithicRunResult, run_monolithic
from abnuts.nuts.predictors import PREDICTOR_MODES

MANIFEST_MATCH_KEYS = (
    "schema_version",
    "command",
    "config_hash",
    "backend",
    "method",
    "model",
    "seed",
    "num_chains",
    "dimension",
    "num_steps",
    "step_size",
    "max_tree_depth",
    "dtype",
    "bucket_size",
    "predictor",
    "save_trace",
)

FLOAT_ATOL = 1e-5
FLOAT_RTOL = 1e-5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the correctness runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=available_models(), required=True)
    parser.add_argument("--num-chains", type=int, required=True)
    parser.add_argument("--dimension", type=int, required=True)
    parser.add_argument("--num-steps", type=int, required=True)
    parser.add_argument("--method", choices=["monolithic", "bucketed", "both"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu"], default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--max-tree-depth", type=int, default=3)
    parser.add_argument("--bucket-size", type=int, default=4)
    parser.add_argument("--predictor", choices=sorted(PREDICTOR_MODES), default="history")
    parser.add_argument("--save-trace", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate correctness-run arguments."""
    for name in ("num_chains", "dimension", "num_steps", "max_tree_depth"):
        value = getattr(args, name)
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive, got {value!r}")
    if args.step_size <= 0.0:
        raise ValueError(f"--step-size must be positive, got {args.step_size!r}")
    if args.method in {"bucketed", "both"} and args.bucket_size <= 0:
        raise ValueError(f"--bucket-size must be positive, got {args.bucket_size!r}")
    if args.backend != "cpu":
        raise ValueError(f"run_correctness currently supports backend='cpu', got {args.backend!r}")


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return a stable JSON-friendly run configuration."""
    config = {
        "backend": args.backend,
        "method": args.method,
        "model": args.model,
        "seed": args.seed,
        "num_chains": args.num_chains,
        "dimension": args.dimension,
        "num_steps": args.num_steps,
        "step_size": args.step_size,
        "max_tree_depth": args.max_tree_depth,
        "dtype": "float32",
    }
    if args.method in {"bucketed", "both"}:
        config["bucket_size"] = args.bucket_size
        config["predictor"] = args.predictor
    config["save_trace"] = bool(args.save_trace)
    return config


def build_manifest(
    args: argparse.Namespace,
    config: dict[str, Any],
    monolithic_result: MonolithicRunResult | None,
    bucketed_result: BucketedRunResult | None,
) -> dict[str, Any]:
    """Build a manifest for a correctness run."""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "command": "python -m abnuts.experiments.run_correctness",
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(args.out),
        "config_hash": stable_json_hash(config),
        **config,
        "outputs": _output_manifest(args.method, save_trace=args.save_trace),
    }
    if monolithic_result is not None:
        manifest["monolithic_final_state"] = _state_shape_manifest(monolithic_result.final_state)
    if bucketed_result is not None:
        manifest["bucketed_final_state"] = _state_shape_manifest(bucketed_result.final_state)
        manifest["bucketed_planner"] = {
            "num_recorded_plans": len(bucketed_result.bucket_plans),
            "final_padding_ratio": float(bucketed_result.bucket_plans[-1].padding_ratio),
        }
    if monolithic_result is not None and bucketed_result is not None:
        manifest["equivalence"] = _equivalence_summary(monolithic_result, bucketed_result)
    return manifest


def write_run_outputs(
    out_dir: Path,
    *,
    method: str,
    monolithic_result: MonolithicRunResult | None,
    bucketed_result: BucketedRunResult | None,
    diagnostic_rows: tuple[Any, ...] = (),
    trace_qa_summary: dict[str, Any] | None = None,
    overwrite: bool,
) -> None:
    """Write deterministic CSV/JSONL outputs for a correctness run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if method == "monolithic":
        if monolithic_result is None:
            raise ValueError("monolithic result is required for method='monolithic'")
        _write_single_run_outputs(
            out_dir,
            monolithic_result,
            prefix="",
            event_name="monolithic_iteration",
            overwrite=overwrite,
        )
        _write_trace_diagnostics(out_dir, diagnostic_rows, overwrite=overwrite)
        return

    if method == "bucketed":
        if bucketed_result is None:
            raise ValueError("bucketed result is required for method='bucketed'")
        _write_single_run_outputs(
            out_dir,
            bucketed_result,
            prefix="",
            event_name="bucketed_iteration",
            overwrite=overwrite,
        )
        _write_bucket_plan_csv(out_dir / "bucket_plan.csv", bucketed_result, overwrite=overwrite)
        _write_trace_diagnostics(out_dir, diagnostic_rows, overwrite=overwrite)
        return

    if monolithic_result is None or bucketed_result is None:
        raise ValueError("both monolithic and bucketed results are required for method='both'")
    _write_single_run_outputs(
        out_dir,
        monolithic_result,
        prefix="monolithic_",
        event_name="monolithic_iteration",
        overwrite=overwrite,
    )
    _write_single_run_outputs(
        out_dir,
        bucketed_result,
        prefix="bucketed_",
        event_name="bucketed_iteration",
        overwrite=overwrite,
    )
    _write_bucket_plan_csv(
        out_dir / "bucketed_bucket_plan.csv",
        bucketed_result,
        overwrite=overwrite,
    )
    _write_text_preserving(
        out_dir / "equivalence.json",
        json.dumps(
            _equivalence_summary(monolithic_result, bucketed_result),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        overwrite=overwrite,
    )
    _write_trace_diagnostics(out_dir, diagnostic_rows, overwrite=overwrite)
    if trace_qa_summary is not None:
        _write_text_preserving(
            out_dir / "trace_qa.json",
            json.dumps(trace_qa_summary, indent=2, sort_keys=True) + "\n",
            overwrite=overwrite,
        )


def _write_trace_diagnostics(
    out_dir: Path,
    diagnostic_rows: tuple[Any, ...],
    *,
    overwrite: bool,
) -> None:
    if not diagnostic_rows:
        return
    _write_text_preserving(
        out_dir / "diagnostics.csv",
        diagnostics_rows_to_csv(diagnostic_rows),
        overwrite=overwrite,
    )


def _write_single_run_outputs(
    out_dir: Path,
    result: MonolithicRunResult | BucketedRunResult,
    *,
    prefix: str,
    event_name: str,
    overwrite: bool,
) -> None:
    _write_text_preserving(
        out_dir / f"{prefix}per_iteration.csv",
        _per_iteration_csv(result),
        overwrite=overwrite,
    )
    _write_text_preserving(
        out_dir / f"{prefix}events.jsonl",
        _events_jsonl(result, event_name=event_name),
        overwrite=overwrite,
    )
    _write_text_preserving(
        out_dir / f"{prefix}positions.csv",
        _positions_csv(result.trace_positions),
        overwrite=overwrite,
    )
    _write_text_preserving(
        out_dir / f"{prefix}final_positions.csv",
        _positions_csv(result.final_state.position[jnp.newaxis, ...]),
        overwrite=overwrite,
    )


def _per_iteration_csv(result: MonolithicRunResult | BucketedRunResult) -> str:
    info = _as_numpy_info(result)
    rows: list[dict[str, Any]] = []
    for step in range(info["acceptance_statistic"].shape[0]):
        rows.append(
            {
                "step": step + 1,
                "mean_acceptance_statistic": float(np.mean(info["acceptance_statistic"][step])),
                "divergence_count": int(np.sum(info["divergence_flag"][step])),
                "total_leapfrog_count": int(np.sum(info["leapfrog_count"][step])),
                "max_realized_tree_depth": int(np.max(info["realized_tree_depth"][step])),
                "mean_gradient_norm": float(np.mean(info["gradient_norm"][step])),
                "max_tree_depth_hit_count": int(np.sum(info["max_tree_depth_hit"][step])),
            }
        )
    return _csv_from_rows(rows)


def _events_jsonl(
    result: MonolithicRunResult | BucketedRunResult,
    *,
    event_name: str,
) -> str:
    info = _as_numpy_info(result)
    lines: list[str] = []
    for step in range(info["acceptance_statistic"].shape[0]):
        event = {
            "event": event_name,
            "step": step + 1,
            "transition_metrics": {
                name: info[name][step].tolist()
                for name in (
                    "acceptance_statistic",
                    "divergence_flag",
                    "realized_tree_depth",
                    "leapfrog_count",
                    "energy_error",
                    "gradient_norm",
                    "max_tree_depth_hit",
                )
            },
        }
        lines.append(json.dumps(event, sort_keys=True))
    return "\n".join(lines) + "\n"


def _positions_csv(positions: Any) -> str:
    position_array = np.asarray(positions)
    rows: list[dict[str, Any]] = []
    for step in range(position_array.shape[0]):
        for chain in range(position_array.shape[1]):
            row: dict[str, Any] = {"step": step + 1, "chain": chain}
            for dim, value in enumerate(position_array[step, chain]):
                row[f"dim_{dim}"] = float(value)
            rows.append(row)
    return _csv_from_rows(rows)


def _csv_from_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _as_numpy_info(result: MonolithicRunResult | BucketedRunResult) -> dict[str, np.ndarray]:
    info = result.transition_info
    return {
        "acceptance_statistic": np.asarray(info.acceptance_statistic),
        "divergence_flag": np.asarray(info.divergence_flag),
        "realized_tree_depth": np.asarray(info.realized_tree_depth),
        "leapfrog_count": np.asarray(info.leapfrog_count),
        "energy_error": np.asarray(info.energy_error),
        "gradient_norm": np.asarray(info.gradient_norm),
        "max_tree_depth_hit": np.asarray(info.max_tree_depth_hit),
    }


def _write_bucket_plan_csv(
    path: Path,
    result: BucketedRunResult,
    *,
    overwrite: bool,
) -> None:
    rows: list[dict[str, Any]] = []
    for step, plan in enumerate(result.bucket_plans, start=1):
        bucket_sizes = np.asarray(plan.bucket_sizes)
        occupancy = np.asarray(plan.occupancy)
        padding = np.asarray(plan.bucket_padding_count)
        for bucket_number in range(plan.num_buckets):
            rows.append(
                {
                    "step": step,
                    "bucket": bucket_number,
                    "bucket_size": int(bucket_sizes[bucket_number]),
                    "occupancy": int(occupancy[bucket_number]),
                    "padding_count": int(padding[bucket_number]),
                }
            )
    _write_text_preserving(path, _csv_from_rows(rows), overwrite=overwrite)


def _state_shape_manifest(state: Any) -> dict[str, list[int]]:
    return {
        "position_shape": list(state.position.shape),
        "potential_energy_shape": list(state.potential_energy.shape),
        "potential_energy_grad_shape": list(state.potential_energy_grad.shape),
    }


def _equivalence_summary(
    monolithic_result: MonolithicRunResult,
    bucketed_result: BucketedRunResult,
) -> dict[str, Any]:
    mono_positions = np.asarray(monolithic_result.trace_positions)
    bucket_positions = np.asarray(bucketed_result.trace_positions)
    position_delta = np.asarray(
        jnp.abs(monolithic_result.trace_positions - bucketed_result.trace_positions)
    )
    mono_info = _as_numpy_info(monolithic_result)
    bucket_info = _as_numpy_info(bucketed_result)
    metrics_equal = {
        name: bool(np.array_equal(bucket_info[name], mono_info[name])) for name in mono_info
    }
    metrics_allclose = {
        name: _metric_matches_with_tolerance(bucket_info[name], mono_info[name])
        for name in mono_info
    }
    metric_max_abs_delta = {
        name: _max_abs_delta(bucket_info[name], mono_info[name])
        for name in mono_info
        if np.issubdtype(mono_info[name].dtype, np.floating)
    }
    metric_mismatch_count = {
        name: int(np.sum(bucket_info[name] != mono_info[name])) for name in mono_info
    }
    float_metrics = tuple(
        name for name, values in mono_info.items() if np.issubdtype(values.dtype, np.floating)
    )
    discrete_metrics = tuple(name for name in mono_info if name not in float_metrics)
    positions_equal = bool(np.array_equal(mono_positions, bucket_positions))
    positions_allclose = bool(
        np.allclose(
            mono_positions,
            bucket_positions,
            atol=FLOAT_ATOL,
            rtol=FLOAT_RTOL,
        )
    )
    final_rng_keys_equal = bool(
        np.array_equal(
            np.asarray(monolithic_result.final_rng_keys),
            np.asarray(bucketed_result.final_rng_keys),
        )
    )
    discrete_metrics_exact = all(metrics_equal[name] for name in discrete_metrics)
    float_metrics_within_tolerance = all(metrics_allclose[name] for name in float_metrics)
    equivalence_passed = (
        positions_allclose
        and final_rng_keys_equal
        and discrete_metrics_exact
        and float_metrics_within_tolerance
    )
    return {
        "equivalence_passed": bool(equivalence_passed),
        "float_atol": FLOAT_ATOL,
        "float_rtol": FLOAT_RTOL,
        "positions_equal": positions_equal,
        "positions_allclose": positions_allclose,
        "max_position_delta": float(np.max(position_delta)) if position_delta.size else 0.0,
        "final_rng_keys_equal": final_rng_keys_equal,
        "discrete_metrics_exact": bool(discrete_metrics_exact),
        "float_metrics_within_tolerance": bool(float_metrics_within_tolerance),
        "metrics_equal": metrics_equal,
        "metrics_allclose": metrics_allclose,
        "metric_max_abs_delta": metric_max_abs_delta,
        "metric_mismatch_count": metric_mismatch_count,
    }


def _metric_matches_with_tolerance(bucket_values: np.ndarray, mono_values: np.ndarray) -> bool:
    if np.issubdtype(mono_values.dtype, np.floating):
        return bool(
            np.allclose(
                bucket_values,
                mono_values,
                atol=FLOAT_ATOL,
                rtol=FLOAT_RTOL,
            )
        )
    return bool(np.array_equal(bucket_values, mono_values))


def _max_abs_delta(bucket_values: np.ndarray, mono_values: np.ndarray) -> float:
    delta = np.abs(bucket_values - mono_values)
    return float(np.max(delta)) if delta.size else 0.0


def _output_manifest(method: str, *, save_trace: bool) -> dict[str, str]:
    if method in {"monolithic", "bucketed"}:
        outputs = {
            "events_jsonl": "events.jsonl",
            "per_iteration_csv": "per_iteration.csv",
            "positions_csv": "positions.csv",
            "final_positions_csv": "final_positions.csv",
        }
        if method == "bucketed":
            outputs["bucket_plan_csv"] = "bucket_plan.csv"
        if save_trace:
            outputs["diagnostics_csv"] = "diagnostics.csv"
        return outputs
    outputs = {
        "monolithic_events_jsonl": "monolithic_events.jsonl",
        "monolithic_per_iteration_csv": "monolithic_per_iteration.csv",
        "monolithic_positions_csv": "monolithic_positions.csv",
        "monolithic_final_positions_csv": "monolithic_final_positions.csv",
        "bucketed_events_jsonl": "bucketed_events.jsonl",
        "bucketed_per_iteration_csv": "bucketed_per_iteration.csv",
        "bucketed_positions_csv": "bucketed_positions.csv",
        "bucketed_final_positions_csv": "bucketed_final_positions.csv",
        "bucketed_bucket_plan_csv": "bucketed_bucket_plan.csv",
        "equivalence_json": "equivalence.json",
    }
    if save_trace:
        outputs["diagnostics_csv"] = "diagnostics.csv"
        outputs["trace_qa_json"] = "trace_qa.json"
    return outputs


def _build_trace_diagnostics(
    monolithic_result: MonolithicRunResult | None,
    bucketed_result: BucketedRunResult | None,
) -> tuple[Any, ...]:
    rows: list[Any] = []
    if monolithic_result is not None:
        rows.extend(
            summarize_trace_diagnostics(
                method="monolithic",
                trace_positions=monolithic_result.trace_positions,
                transition_info=monolithic_result.transition_info,
            )
        )
    if bucketed_result is not None:
        rows.extend(
            summarize_trace_diagnostics(
                method="bucketed",
                trace_positions=bucketed_result.trace_positions,
                transition_info=bucketed_result.transition_info,
            )
        )
    return tuple(rows)


def _trace_qa_summary(
    monolithic_result: MonolithicRunResult,
    bucketed_result: BucketedRunResult,
    diagnostic_rows: tuple[Any, ...],
) -> dict[str, Any]:
    equivalence = _equivalence_summary(monolithic_result, bucketed_result)
    midpoint = len(diagnostic_rows) // 2
    diagnostics_equal = diagnostics_rows_equal(
        diagnostic_rows[:midpoint],
        diagnostic_rows[midpoint:],
    )
    summary = {
        **equivalence,
        "saved_positions_equal": equivalence["positions_equal"],
        "saved_positions_allclose": equivalence["positions_allclose"],
        "diagnostics_equal": diagnostics_equal,
    }
    if not summary["saved_positions_allclose"]:
        raise ValueError(
            "trace-level QA failed: saved monolithic and bucketed positions exceed "
            "strict tolerance"
        )
    if not summary["equivalence_passed"]:
        raise ValueError(
            "trace-level QA failed: monolithic and bucketed outputs exceed strict "
            "tolerance or discrete metrics differ"
        )
    if not diagnostics_equal:
        raise ValueError("trace-level QA failed: monolithic and bucketed diagnostics differ")
    return summary


def _write_text_preserving(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        raise FileExistsError(
            f"Output file already exists with different content: {path}. "
            "Pass --overwrite to replace it intentionally."
        )
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the correctness command."""
    args = parse_args(argv)
    try:
        validate_args(args)
        config = build_run_config(args)
        model = get_model(args.model, dimension=args.dimension)
        initial_positions = model.initial_position(
            key=args.seed,
            num_chains=args.num_chains,
        )
        monolithic_result = None
        bucketed_result = None
        if args.method in {"monolithic", "both"}:
            monolithic_result = run_monolithic(
                model,
                initial_positions,
                jr.PRNGKey(args.seed),
                num_steps=args.num_steps,
                step_size=args.step_size,
                max_tree_depth=args.max_tree_depth,
                dtype=jnp.float32,
            )
        if args.method in {"bucketed", "both"}:
            bucketed_result = run_bucketed(
                model,
                initial_positions,
                jr.PRNGKey(args.seed),
                num_steps=args.num_steps,
                step_size=args.step_size,
                max_tree_depth=args.max_tree_depth,
                bucket_size=args.bucket_size,
                predictor=args.predictor,
                dtype=jnp.float32,
            )
        monolithic_result, bucketed_result = block_until_ready_tree(
            (monolithic_result, bucketed_result)
        )
        if monolithic_result is not None and bucketed_result is not None:
            equivalence = _equivalence_summary(monolithic_result, bucketed_result)
            if not equivalence["equivalence_passed"]:
                raise ValueError(
                    "monolithic-vs-bucketed equivalence failed strict gate: "
                    f"{json.dumps(equivalence, sort_keys=True)}"
                )
        diagnostic_rows: tuple[Any, ...] = ()
        trace_qa_summary = None
        if args.save_trace:
            diagnostic_rows = _build_trace_diagnostics(monolithic_result, bucketed_result)
            if monolithic_result is not None and bucketed_result is not None:
                trace_qa_summary = _trace_qa_summary(
                    monolithic_result,
                    bucketed_result,
                    diagnostic_rows,
                )
        manifest = build_manifest(args, config, monolithic_result, bucketed_result)
        manifest_path, wrote_manifest = write_manifest(
            args.out,
            manifest,
            overwrite=args.overwrite,
            comparable_keys=MANIFEST_MATCH_KEYS,
        )
        write_run_outputs(
            args.out,
            method=args.method,
            monolithic_result=monolithic_result,
            bucketed_result=bucketed_result,
            diagnostic_rows=diagnostic_rows,
            trace_qa_summary=trace_qa_summary,
            overwrite=args.overwrite,
        )
    except (DiagnosticsDependencyError, OSError, ValueError) as exc:
        print(
            "Correctness run failed "
            f"(backend={getattr(args, 'backend', 'unknown')}, "
            f"method={getattr(args, 'method', 'unknown')}, "
            f"model={getattr(args, 'model', 'unknown')}, "
            f"out={getattr(args, 'out', 'unknown')}): {exc}",
            file=sys.stderr,
        )
        return 2

    if wrote_manifest:
        print(f"Wrote correctness outputs: {args.out}")
    else:
        print(f"Preserved existing correctness outputs: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
