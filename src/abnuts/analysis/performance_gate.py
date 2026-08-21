"""Generate the T37 performance gate report from existing repair artifacts.

The gate is a decision report, not a benchmark runner. It reads raw outputs from
T32-T36, writes a compact processed CSV plus human-readable reports, and states
whether broad paper sweeps are scientifically justified.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


COMMAND = "python -m abnuts.analysis.performance_gate"
DEFAULT_PRE_REPAIR_RUN_DIR = Path("/home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw")
DEFAULT_PRE_REPAIR_FORENSICS_CSV = Path(
    "results/processed/performance_forensics/forensics_summary.csv"
)
DEFAULT_CORRECTNESS_DIR = Path("results/raw/correctness/gp_tiny_repair")
DEFAULT_HETEROGENEOUS_SUMMARIES = (
    Path("results/raw/t34_fixed_shape_bucket_executor/tiny_cpu/summary.csv"),
    Path("results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling/summary.csv"),
)
DEFAULT_ORACLE_GAP_DIR = Path("results/raw/oracle_gap/repair_tiny")
DEFAULT_MIN_HETEROGENEOUS_SPEEDUP = 1.05
DEFAULT_BUCKET_LOOP_MAX_RUNTIME_RATIO = 1.25
DEFAULT_MIN_NEGATIVE_CONTROL_SPEEDUP = 0.80
# Executed lane-steps relative to one undifferentiated batch. Anything at or
# above 1.0 means bucketing is not reclaiming any straggler work at all.
DEFAULT_MAX_EXECUTED_WORK_RATIO = 1.0


@dataclass(frozen=True)
class GateCriterion:
    """One pass/fail line in the performance gate."""

    gate: str
    status: str
    threshold: str
    value: str
    evidence: str
    source: str


@dataclass(frozen=True)
class SpeedupCandidate:
    """One repaired bucketed speedup candidate."""

    label: str
    speedup: float
    source: Path
    is_analysis_upper_bound: bool = False


@dataclass(frozen=True)
class MechanismEvidence:
    """Executed-work evidence that bucketing reclaims straggler waste.

    Wall-clock speedup mixes the mechanism with hardware, planner cost, and
    predictor quality. Executed lane-steps isolate the mechanism itself and are
    hardware independent, so this is reported alongside — never instead of — the
    wall-clock gate.
    """

    best_label: str
    executed_work_ratio: float
    oracle_plan_work_ratio: float
    predictor_abs_error: float | None
    planner_share: float | None
    oracle_current_speedup: float | None
    source: Path


@dataclass(frozen=True)
class GateReport:
    """Computed T37 gate result and report metadata."""

    status: str
    criteria: tuple[GateCriterion, ...]
    pre_repair_bucketed_rows: int
    pre_repair_faster_rows: int
    pre_repair_best_speedup: float | None
    pre_repair_median_speedup: float | None
    pre_repair_gp_mismatches: int
    pre_repair_gp_bucketed_rows: int
    repaired_candidate_rows: int
    repaired_faster_rows: int
    repaired_best: SpeedupCandidate | None
    repaired_best_non_analysis: SpeedupCandidate | None
    oracle_current_best: SpeedupCandidate | None
    bucket_loop_runtime_ratio: float | None
    bucket_loop_rows: tuple[dict[str, str], ...]
    mechanism: MechanismEvidence | None
    next_task_hint: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pre-repair-run-dir",
        type=Path,
        default=DEFAULT_PRE_REPAIR_RUN_DIR,
        help="Raw run-21070 tree containing summary.csv files.",
    )
    parser.add_argument(
        "--pre-repair-forensics-csv",
        type=Path,
        default=DEFAULT_PRE_REPAIR_FORENSICS_CSV,
        help="Fallback T32 forensics CSV generated from raw run-21070 outputs.",
    )
    parser.add_argument(
        "--correctness-dir",
        type=Path,
        default=DEFAULT_CORRECTNESS_DIR,
        help="T35 correctness output directory containing equivalence.json.",
    )
    parser.add_argument(
        "--heterogeneous-summary",
        type=Path,
        action="append",
        default=None,
        help=(
            "T34 repaired heterogeneous benchmark summary.csv. May be repeated. "
            "Defaults to the repaired tiny CPU summaries."
        ),
    )
    parser.add_argument(
        "--oracle-gap-dir",
        type=Path,
        default=DEFAULT_ORACLE_GAP_DIR,
        help="T36 oracle-gap raw output directory containing summary.csv.",
    )
    parser.add_argument(
        "--negative-control-summary",
        type=Path,
        action="append",
        default=None,
        help="Optional homogeneous negative-control summary.csv. May be repeated.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Processed output directory for gate_summary.csv, gate_report.md, and manifest.json.",
    )
    parser.add_argument(
        "--latex-out",
        type=Path,
        default=None,
        help="Optional LaTeX output directory for gate_report.tex.",
    )
    parser.add_argument(
        "--min-heterogeneous-speedup",
        type=float,
        default=DEFAULT_MIN_HETEROGENEOUS_SPEEDUP,
        help="Required repaired non-analysis heterogeneous warm speedup.",
    )
    parser.add_argument(
        "--bucket-loop-max-runtime-ratio",
        type=float,
        default=DEFAULT_BUCKET_LOOP_MAX_RUNTIME_RATIO,
        help="Maximum repaired warm runtime max/min ratio across bucket-count rows.",
    )
    parser.add_argument(
        "--min-negative-control-speedup",
        type=float,
        default=DEFAULT_MIN_NEGATIVE_CONTROL_SPEEDUP,
        help="Minimum acceptable homogeneous negative-control speedup if provided.",
    )
    parser.add_argument(
        "--max-executed-work-ratio",
        type=float,
        default=DEFAULT_MAX_EXECUTED_WORK_RATIO,
        help="Maximum non-analysis executed-work ratio for the mechanism gate.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing report files.")
    return parser.parse_args(argv)


def build_gate_report(
    *,
    pre_repair_run_dir: Path,
    pre_repair_forensics_csv: Path,
    correctness_dir: Path,
    heterogeneous_summaries: tuple[Path, ...],
    oracle_gap_dir: Path,
    negative_control_summaries: tuple[Path, ...],
    min_heterogeneous_speedup: float = DEFAULT_MIN_HETEROGENEOUS_SPEEDUP,
    bucket_loop_max_runtime_ratio: float = DEFAULT_BUCKET_LOOP_MAX_RUNTIME_RATIO,
    min_negative_control_speedup: float = DEFAULT_MIN_NEGATIVE_CONTROL_SPEEDUP,
    max_executed_work_ratio: float = DEFAULT_MAX_EXECUTED_WORK_RATIO,
) -> GateReport:
    """Compute all T37 gate criteria from on-disk artifacts."""
    pre_rows, pre_source = _load_pre_repair_rows(pre_repair_run_dir, pre_repair_forensics_csv)
    pre_stats = _pre_repair_stats(pre_rows)

    correctness = _load_correctness(correctness_dir)
    correctness_passed, correctness_value, correctness_note = _evaluate_correctness(correctness)

    repaired_candidates = _load_repaired_candidates(heterogeneous_summaries, oracle_gap_dir)
    non_analysis = [candidate for candidate in repaired_candidates if not candidate.is_analysis_upper_bound]
    repaired_best = _best_candidate(repaired_candidates)
    repaired_best_non_analysis = _best_candidate(non_analysis)
    repaired_faster_rows = sum(1 for candidate in non_analysis if candidate.speedup > 1.0)
    heterogeneous_passed = (
        repaired_best_non_analysis is not None
        and repaired_best_non_analysis.speedup >= min_heterogeneous_speedup
    )

    bucket_loop_passed, bucket_loop_value, bucket_loop_note, bucket_loop_ratio, bucket_loop_rows = (
        _evaluate_bucket_loop_scaling(heterogeneous_summaries, bucket_loop_max_runtime_ratio)
    )

    negative_passed, negative_value, negative_note, negative_source = _evaluate_negative_controls(
        negative_control_summaries,
        min_negative_control_speedup,
    )

    mechanism_passed, mechanism_value, mechanism_note, mechanism = _evaluate_mechanism(
        oracle_gap_dir,
        max_executed_work_ratio,
    )

    generated_from_raw = (
        all(path.exists() for path in heterogeneous_summaries)
        and (oracle_gap_dir / "summary.csv").exists()
        and all(path.exists() for path in negative_control_summaries)
    )

    criteria = (
        GateCriterion(
            gate="correctness",
            status=_status(correctness_passed),
            threshold="equivalence_passed and exact discrete/RNG metrics",
            value=correctness_value,
            evidence=correctness_note,
            source=str(correctness_dir / "equivalence.json"),
        ),
        GateCriterion(
            gate="mechanism_executed_work",
            status=_status(mechanism_passed),
            threshold=(
                "non-analysis executed-work ratio < "
                f"{max_executed_work_ratio:.2f} (hardware independent)"
            ),
            value=mechanism_value,
            evidence=mechanism_note,
            source=str(oracle_gap_dir / "summary.csv"),
        ),
        GateCriterion(
            gate="heterogeneous_speedup",
            status=_status(heterogeneous_passed),
            threshold=f"non-analysis repaired speedup >= {min_heterogeneous_speedup:.2f}x",
            value=(
                f"{repaired_best_non_analysis.speedup:.3f}x"
                if repaired_best_non_analysis is not None
                else "unavailable"
            ),
            evidence=(
                f"Best non-analysis repaired row: {repaired_best_non_analysis.label}"
                if repaired_best_non_analysis is not None
                else "No repaired non-analysis bucketed rows were found."
            ),
            source=", ".join(str(path) for path in heterogeneous_summaries)
            + f"; {oracle_gap_dir / 'summary.csv'}",
        ),
        GateCriterion(
            gate="homogeneous_negative_control",
            status=_status(negative_passed),
            threshold=(
                "homogeneous control present, slowdown explained, "
                f"speedup >= {min_negative_control_speedup:.2f}x"
            ),
            value=negative_value,
            evidence=negative_note,
            source=negative_source,
        ),
        GateCriterion(
            gate="python_bucket_loop_overhead",
            status=_status(bucket_loop_passed),
            threshold=f"repaired warm runtime max/min ratio <= {bucket_loop_max_runtime_ratio:.2f}",
            value=bucket_loop_value,
            evidence=bucket_loop_note,
            source=", ".join(str(path) for path in heterogeneous_summaries),
        ),
        GateCriterion(
            gate="report_from_raw_results",
            status=_status(generated_from_raw),
            threshold="all named raw summaries are present",
            value="present" if generated_from_raw else "missing",
            evidence=(
                "Gate uses repaired raw summaries plus T35 equivalence JSON; "
                f"T32 pre-repair source: {pre_source}."
            ),
            source=", ".join(
                [
                    str(pre_source),
                    str(correctness_dir / "equivalence.json"),
                    *(str(path) for path in heterogeneous_summaries),
                    str(oracle_gap_dir / "summary.csv"),
                    *(str(path) for path in negative_control_summaries),
                ]
            ),
        ),
        GateCriterion(
            gate="faster_rows_reported",
            status="PASS",
            threshold="report includes bucketed faster-row counts",
            value=(
                f"pre-repair {pre_stats['n_faster']}/{pre_stats['n_bucketed']}; "
                f"repaired {repaired_faster_rows}/{len(non_analysis)}"
            ),
            evidence="Counts are computed directly from speedup columns.",
            source=f"{pre_source}; repaired summaries",
        ),
    )

    required_passed = all(
        criterion.status == "PASS"
        for criterion in criteria
        if criterion.gate != "faster_rows_reported"
    )
    status = "PASS" if required_passed else "FAIL"
    if status == "PASS":
        next_task_hint = "Promote T38 minimal HPC validation."
    elif mechanism_passed and not heterogeneous_passed:
        # The mechanism works but the deployed scheduler does not exploit it.
        # Sending this to HPC would only buy a faster measurement of the same
        # predictor gap, so name the actual blocker instead.
        next_task_hint = (
            "Do not promote T38. Mechanism present, wall-clock gate not met: the executor "
            "reclaims work but the deployed predictor does not find it. The next task should "
            "repair predictor quality and planner cost, not run HPC validation."
        )
    else:
        next_task_hint = (
            "Do not promote T38. Add another repair task for the remaining failed gates."
        )

    return GateReport(
        status=status,
        criteria=criteria,
        pre_repair_bucketed_rows=int(pre_stats["n_bucketed"]),
        pre_repair_faster_rows=int(pre_stats["n_faster"]),
        pre_repair_best_speedup=pre_stats["best"],
        pre_repair_median_speedup=pre_stats["median"],
        pre_repair_gp_mismatches=int(pre_stats["gp_mismatches"]),
        pre_repair_gp_bucketed_rows=int(pre_stats["gp_bucketed"]),
        repaired_candidate_rows=len(non_analysis),
        repaired_faster_rows=repaired_faster_rows,
        repaired_best=repaired_best,
        repaired_best_non_analysis=repaired_best_non_analysis,
        oracle_current_best=_best_candidate(
            [candidate for candidate in repaired_candidates if candidate.is_analysis_upper_bound]
        ),
        bucket_loop_runtime_ratio=bucket_loop_ratio,
        bucket_loop_rows=tuple(bucket_loop_rows),
        mechanism=mechanism,
        next_task_hint=next_task_hint,
    )


def write_outputs(
    report: GateReport,
    *,
    out_dir: Path,
    latex_out_dir: Path | None,
    sources: dict[str, Any],
    overwrite: bool,
) -> tuple[Path, Path, Path, Path | None]:
    """Write processed CSV, markdown, manifest, and optional LaTeX output."""
    _prepare_output_dir(out_dir, overwrite=overwrite)
    summary_path = out_dir / "gate_summary.csv"
    markdown_path = out_dir / "gate_report.md"
    manifest_path = out_dir / "manifest.json"
    summary_path.write_text(_criteria_csv(report.criteria), encoding="utf-8")
    markdown_path.write_text(_markdown_report(report), encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "command": COMMAND,
                "gate_status": report.status,
                "sources": _jsonable_sources(sources),
                "outputs": {
                    "gate_summary_csv": str(summary_path),
                    "gate_report_md": str(markdown_path),
                    "gate_report_tex": str(latex_out_dir / "gate_report.tex")
                    if latex_out_dir is not None
                    else None,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    latex_path: Path | None = None
    if latex_out_dir is not None:
        _prepare_output_dir(latex_out_dir, overwrite=overwrite)
        latex_path = latex_out_dir / "gate_report.tex"
        latex_path.write_text(_latex_report(report), encoding="utf-8")

    return summary_path, markdown_path, manifest_path, latex_path


def _load_pre_repair_rows(
    run_dir: Path,
    forensics_csv: Path,
) -> tuple[list[dict[str, str]], Path]:
    if run_dir.exists():
        summary_files = sorted(run_dir.rglob("summary.csv"))
        if summary_files:
            rows: list[dict[str, str]] = []
            for summary in summary_files:
                rel = summary.relative_to(run_dir)
                family = rel.parts[0] if len(rel.parts) > 1 else "unknown"
                for row in _read_csv(summary):
                    row["_family"] = family
                    row["_source_csv"] = str(summary)
                    rows.append(row)
            return rows, run_dir
    if forensics_csv.exists():
        return _read_csv(forensics_csv), forensics_csv
    raise FileNotFoundError(
        f"expected pre-repair run dir at {run_dir} or forensics CSV at {forensics_csv}"
    )


def _pre_repair_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    bucketed = [row for row in rows if row.get("method") == "bucketed"]
    speedups = [
        speedup
        for speedup in (_row_speedup(row) for row in bucketed)
        if speedup is not None
    ]
    gp_bucketed = [
        row
        for row in bucketed
        if row.get("_family") == "gaussian_process"
        or row.get("model") == "gaussian_process"
        or row.get("model_family") == "gaussian_process"
    ]
    gp_mismatches = 0
    for row in gp_bucketed:
        mono_lf = _as_int(row.get("mono_total_leapfrog_count"))
        bucket_lf = _as_int(row.get("bucket_total_leapfrog_count"))
        if mono_lf is not None and bucket_lf is not None and mono_lf != bucket_lf:
            gp_mismatches += 1
    return {
        "n_bucketed": len(bucketed),
        "n_faster": sum(1 for speedup in speedups if speedup > 1.0),
        "best": max(speedups) if speedups else None,
        "median": statistics.median(speedups) if speedups else None,
        "gp_mismatches": gp_mismatches,
        "gp_bucketed": len(gp_bucketed),
    }


def _load_correctness(correctness_dir: Path) -> dict[str, Any]:
    path = correctness_dir / "equivalence.json"
    if not path.exists():
        raise FileNotFoundError(f"expected correctness equivalence JSON at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_correctness(equivalence: dict[str, Any]) -> tuple[bool, str, str]:
    required = {
        "equivalence_passed": _as_bool(equivalence.get("equivalence_passed")),
        "final_rng_keys_equal": _as_bool(equivalence.get("final_rng_keys_equal")),
        "discrete_metrics_exact": _as_bool(equivalence.get("discrete_metrics_exact")),
        "float_metrics_within_tolerance": _as_bool(
            equivalence.get("float_metrics_within_tolerance")
        ),
        "positions_allclose": _as_bool(equivalence.get("positions_allclose")),
    }
    passed = all(required.values())
    max_position_delta = equivalence.get("max_position_delta", "unavailable")
    metric_max_abs_delta = equivalence.get("metric_max_abs_delta", {})
    value = "; ".join(f"{key}={value}" for key, value in required.items())
    note = (
        f"max_position_delta={max_position_delta}; "
        f"metric_max_abs_delta={metric_max_abs_delta}"
    )
    return passed, value, note


def _load_repaired_candidates(
    heterogeneous_summaries: tuple[Path, ...],
    oracle_gap_dir: Path,
) -> list[SpeedupCandidate]:
    candidates: list[SpeedupCandidate] = []
    for summary in heterogeneous_summaries:
        rows = _read_csv(summary)
        for row in rows:
            if row.get("method") != "bucketed":
                continue
            speedup = _row_speedup(row)
            if speedup is None:
                continue
            candidates.append(
                SpeedupCandidate(
                    label=_benchmark_label(row),
                    speedup=speedup,
                    source=summary,
                    is_analysis_upper_bound=False,
                )
            )

    oracle_summary = oracle_gap_dir / "summary.csv"
    for row in _read_csv(oracle_summary):
        if row.get("scheduler_mode") == "monolithic":
            continue
        speedup = _row_speedup(row)
        if speedup is None:
            continue
        is_upper = _as_bool(row.get("is_analysis_upper_bound"))
        candidates.append(
            SpeedupCandidate(
                label=_oracle_label(row),
                speedup=speedup,
                source=oracle_summary,
                is_analysis_upper_bound=is_upper,
            )
        )
    return candidates


def _evaluate_bucket_loop_scaling(
    heterogeneous_summaries: tuple[Path, ...],
    max_runtime_ratio: float,
) -> tuple[bool, str, str, float | None, list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    for summary in heterogeneous_summaries:
        for row in _read_csv(summary):
            if row.get("method") == "bucketed" and _as_int(row.get("bucket_num_buckets")):
                copied = dict(row)
                copied["_source_csv"] = str(summary)
                rows.append(copied)

    no_padding_rows = [
        row for row in rows if (_as_int(row.get("bucket_padding_count")) or 0) == 0
    ]
    by_bucket_count: dict[int, list[float]] = {}
    for row in no_padding_rows:
        num_buckets = _as_int(row.get("bucket_num_buckets"))
        warm = _row_warm_time(row)
        if num_buckets is None or warm is None:
            continue
        by_bucket_count.setdefault(num_buckets, []).append(warm)

    if len(by_bucket_count) < 2:
        return (
            False,
            "unavailable",
            "Need at least two repaired no-padding bucket-count rows.",
            None,
            rows,
        )

    medians = {
        num_buckets: statistics.median(values)
        for num_buckets, values in sorted(by_bucket_count.items())
    }
    min_warm = min(medians.values())
    max_warm = max(medians.values())
    ratio = max_warm / min_warm if min_warm > 0 else float("inf")
    passed = ratio <= max_runtime_ratio
    value = f"{ratio:.3f}"
    note = (
        "No-padding repaired warm medians by bucket count: "
        + ", ".join(f"{num_buckets} buckets={warm:.6f}s" for num_buckets, warm in medians.items())
    )
    return passed, value, note, ratio, rows


def _evaluate_negative_controls(
    negative_control_summaries: tuple[Path, ...],
    min_speedup: float,
) -> tuple[bool, str, str, str]:
    if not negative_control_summaries:
        return (
            False,
            "missing",
            "No raw homogeneous negative-control summary was provided in T32-T36 outputs.",
            "unavailable",
        )

    bucketed_rows: list[tuple[Path, dict[str, str]]] = []
    for summary in negative_control_summaries:
        for row in _read_csv(summary):
            if row.get("method") == "bucketed":
                bucketed_rows.append((summary, row))

    speedups = [
        speedup
        for _, row in bucketed_rows
        for speedup in [_negative_control_speedup(row)]
        if speedup is not None
    ]
    if not speedups:
        return (
            False,
            "unavailable",
            "Homogeneous negative-control summaries contain no bucketed speedup rows.",
            ", ".join(str(path) for path in negative_control_summaries),
        )

    bounded = min(speedups) >= min_speedup
    explained = all(
        _as_bool(row.get("timing_breakdown_enabled"))
        and (
            row.get("diagnostic_conclusion", "") != ""
            or _as_float(row.get("timing_component_jit_measurement_overhead_seconds"))
            is not None
        )
        for _, row in bucketed_rows
    )
    passed = bounded and explained
    profiled_speedups = [
        speedup
        for _, row in bucketed_rows
        for speedup in [_as_float(row.get("speedup_profiled_vs_monolithic"))]
        if speedup is not None
    ]
    component_measurement = [
        seconds
        for _, row in bucketed_rows
        for seconds in [_as_float(row.get("timing_component_jit_measurement_overhead_seconds"))]
        if seconds is not None
    ]
    unattributed = [
        seconds
        for _, row in bucketed_rows
        for seconds in [_as_float(row.get("timing_unattributed_seconds"))]
        if seconds is not None
    ]
    planner = [
        seconds
        for _, row in bucketed_rows
        for seconds in [_as_float(row.get("timing_repeated_planning_seconds"))]
        if seconds is not None
    ]
    primary = sorted(
        {
            row.get("diagnostic_primary_overhead", "")
            for _, row in bucketed_rows
            if row.get("diagnostic_primary_overhead", "")
        }
    )
    note_parts = [
        f"worst production speedup={min(speedups):.3f}x",
        f"best production speedup={max(speedups):.3f}x",
    ]
    if profiled_speedups:
        note_parts.append(f"worst profiled speedup={min(profiled_speedups):.3f}x")
        note_parts.append(f"best profiled speedup={max(profiled_speedups):.3f}x")
    note = "; ".join(note_parts) + "; "
    note += (
        f"max component-measurement overhead={max(component_measurement):.6f}s; "
        if component_measurement
        else "component-measurement overhead unavailable; "
    )
    note += (
        f"max unattributed={max(unattributed):.6f}s; "
        if unattributed
        else "unattributed unavailable; "
    )
    note += (
        f"max repeated planning={max(planner):.6f}s; "
        if planner
        else "repeated planning unavailable; "
    )
    note += (
        f"primary overhead={', '.join(primary)}; "
        if primary
        else "primary overhead unavailable; "
    )
    note += f"timing breakdown and diagnostic explanation present for all bucketed rows={explained}"
    return (
        passed,
        f"{min(speedups):.3f}x",
        note,
        ", ".join(str(path) for path in negative_control_summaries),
    )


def _evaluate_mechanism(
    oracle_gap_dir: Path,
    max_executed_work_ratio: float,
) -> tuple[bool, str, str, MechanismEvidence | None]:
    """Check whether bucketing executes less work than one undifferentiated batch.

    Reads ``executed_work_ratio`` from the oracle-gap raw summary. The ratio is
    lane-steps under the model documented in ``run_oracle_gap``: a vmapped group
    costs its slowest member, so a group's cost is its width times its maximum
    leapfrog count.
    """
    summary_path = oracle_gap_dir / "summary.csv"
    if not summary_path.exists():
        return False, "unavailable", f"No oracle-gap summary at {summary_path}.", None

    rows = _read_csv(summary_path)
    scheduled = [
        row
        for row in rows
        if row.get("scheduler_mode") != "monolithic" and row.get("executed_work_ratio")
    ]
    if not scheduled:
        return (
            False,
            "unavailable",
            (
                "Oracle-gap summary has no executed_work_ratio column. Regenerate it "
                "with the post-T40 runner so the mechanism can be checked from raw results."
            ),
            None,
        )

    non_analysis = [row for row in scheduled if not _as_bool(row.get("is_analysis_upper_bound"))]
    if not non_analysis:
        return False, "unavailable", "No non-analysis scheduled rows were found.", None

    best = min(non_analysis, key=lambda row: _as_float(row.get("executed_work_ratio")) or 1.0)
    ratio = _as_float(best.get("executed_work_ratio"))
    if ratio is None:
        return False, "unavailable", "executed_work_ratio could not be parsed.", None

    oracle_ratio = _as_float(best.get("oracle_plan_executed_work_ratio"))
    planner_seconds = _as_float(best.get("timing_planner_seconds"))
    warm_seconds = _as_float(best.get("timing_warm_iteration_seconds"))
    planner_share = (
        planner_seconds / warm_seconds
        if planner_seconds is not None and warm_seconds
        else None
    )
    oracle_current_rows = [
        row for row in scheduled if _as_bool(row.get("is_analysis_upper_bound"))
    ]
    oracle_current_speedup = (
        max(
            (_as_float(row.get("speedup")) or 0.0 for row in oracle_current_rows),
            default=None,
        )
        if oracle_current_rows
        else None
    )

    evidence = MechanismEvidence(
        best_label=_oracle_label(best),
        executed_work_ratio=ratio,
        oracle_plan_work_ratio=oracle_ratio if oracle_ratio is not None else ratio,
        predictor_abs_error=_as_float(best.get("mean_predictor_abs_error")),
        planner_share=planner_share,
        oracle_current_speedup=oracle_current_speedup,
        source=summary_path,
    )

    passed = ratio < max_executed_work_ratio
    note_parts = [
        f"Best non-analysis executed-work ratio {ratio:.3f} from {evidence.best_label}",
        f"oracle plans on the same realized depths reach {evidence.oracle_plan_work_ratio:.3f}",
    ]
    if evidence.predictor_abs_error is not None:
        note_parts.append(f"mean predictor abs error {evidence.predictor_abs_error:.3f}")
    if planner_share is not None:
        note_parts.append(f"planner is {planner_share * 100:.1f}% of warm time")
    if oracle_current_speedup is not None:
        note_parts.append(
            f"oracle_current wall speedup {oracle_current_speedup:.3f}x (analysis-only)"
        )
    return passed, f"{ratio:.3f}", "; ".join(note_parts) + ".", evidence


def _negative_control_speedup(row: dict[str, str]) -> float | None:
    for key in ("speedup_unprofiled_vs_monolithic", "speedup_vs_monolithic", "speedup"):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _best_candidate(candidates: list[SpeedupCandidate]) -> SpeedupCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.speedup)


def _row_speedup(row: dict[str, str]) -> float | None:
    for key in ("speedup", "speedup_vs_monolithic"):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _row_warm_time(row: dict[str, str]) -> float | None:
    for key in ("timing_warm_iteration_seconds", "time_seconds", "t_bucket", "t_mode"):
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _benchmark_label(row: dict[str, str]) -> str:
    model = row.get("model", "unknown")
    predictor = row.get("predictor", "unknown")
    bucket_size = row.get("bucket_size", "unknown")
    chains = row.get("num_chains", "unknown")
    dimension = row.get("dimension", "unknown")
    buckets = row.get("bucket_num_buckets", "unknown")
    padding = row.get("bucket_padding_ratio", "unknown")
    return (
        f"{model}, predictor={predictor}, bucket_size={bucket_size}, "
        f"C={chains}, D={dimension}, num_buckets={buckets}, padding={padding}"
    )


def _oracle_label(row: dict[str, str]) -> str:
    mode = row.get("scheduler_mode", "unknown")
    label = row.get("scheduler_label", mode)
    bucket_size = row.get("bucket_size", "unknown")
    analysis = " analysis-only" if _as_bool(row.get("is_analysis_upper_bound")) else ""
    return f"oracle-gap {label}, bucket_size={bucket_size}{analysis}"


def _criteria_csv(criteria: tuple[GateCriterion, ...]) -> str:
    buf = StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=("gate", "status", "threshold", "value", "evidence", "source"),
        lineterminator="\n",
    )
    writer.writeheader()
    for criterion in criteria:
        writer.writerow(
            {
                "gate": criterion.gate,
                "status": criterion.status,
                "threshold": criterion.threshold,
                "value": criterion.value,
                "evidence": criterion.evidence,
                "source": criterion.source,
            }
        )
    return buf.getvalue()


def _markdown_report(report: GateReport) -> str:
    lines = [
        "# Performance Gate Before Paper Sweeps",
        "",
        f"**Performance Gate: {report.status}**",
        "",
        "This report is generated from the raw artifacts named under Sources below. It does "
        "not run a broad sweep.",
        "",
        "## Decision",
        "",
    ]
    if report.status == "PASS":
        lines.append("Broad sweeps may proceed to the minimal HPC validation task.")
    else:
        lines.append(
            "Broad sweeps remain blocked. The next active task must be a repair task, not T38."
        )
    lines.extend(["", f"Next task guidance: {report.next_task_hint}", ""])

    lines.extend(
        [
            "## Gate Criteria",
            "",
            "| Gate | Status | Threshold | Value | Evidence |",
            "|------|--------|-----------|-------|----------|",
        ]
    )
    for criterion in report.criteria:
        lines.append(
            "| "
            + " | ".join(
                [
                    criterion.gate,
                    criterion.status,
                    criterion.threshold,
                    criterion.value,
                    criterion.evidence,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Faster Rows",
            "",
            (
                "Pre-repair run 21070 bucketed rows faster than monolithic: "
                f"{report.pre_repair_faster_rows} / {report.pre_repair_bucketed_rows}."
            ),
            (
                "Repaired local non-analysis bucketed rows faster than monolithic: "
                f"{report.repaired_faster_rows} / {report.repaired_candidate_rows}."
            ),
        ]
    )
    if report.pre_repair_best_speedup is not None and report.pre_repair_median_speedup is not None:
        lines.append(
            "Pre-repair speedups are negative diagnostic evidence only: "
            f"best {report.pre_repair_best_speedup:.3f}x, "
            f"median {report.pre_repair_median_speedup:.3f}x."
        )
    if report.repaired_best_non_analysis is not None:
        lines.append(
            "Best repaired non-analysis candidate: "
            f"{report.repaired_best_non_analysis.speedup:.3f}x "
            f"({report.repaired_best_non_analysis.label})."
        )
    if report.oracle_current_best is not None:
        lines.append(
            "Best oracle-current analysis-only upper bound: "
            f"{report.oracle_current_best.speedup:.3f}x "
            f"({report.oracle_current_best.label})."
        )

    lines.extend(["", "## Mechanism Versus Scheduler", ""])
    if report.mechanism is None:
        lines.append(
            "Executed-work evidence was unavailable. Regenerate the oracle-gap raw "
            "summary with the post-T40 runner so the mechanism can be separated from "
            "wall-clock timing."
        )
    else:
        mechanism = report.mechanism
        lines.extend(
            [
                (
                    "Executed lane-steps are hardware independent, so they separate three "
                    "things wall-clock timing mixes together: whether the executor reclaims "
                    "straggler work, whether the scheduler finds that work, and what "
                    "planning costs."
                ),
                "",
                "| quantity | value | reading |",
                "|---|---|---|",
                (
                    f"| deployed scheduler executed-work ratio | {mechanism.executed_work_ratio:.3f} "
                    f"| what {mechanism.best_label} actually reclaimed |"
                ),
                (
                    "| oracle-plan executed-work ratio | "
                    f"{mechanism.oracle_plan_work_ratio:.3f} "
                    "| what the same executor reclaims given perfect grouping |"
                ),
            ]
        )
        if mechanism.predictor_abs_error is not None:
            lines.append(
                f"| mean predictor absolute error | {mechanism.predictor_abs_error:.3f} "
                "| realized-depth prediction error driving the gap above |"
            )
        if mechanism.planner_share is not None:
            lines.append(
                f"| planner share of warm time | {mechanism.planner_share * 100:.1f}% "
                "| scheduling cost paid out of the reclaimed budget |"
            )
        if mechanism.oracle_current_speedup is not None:
            lines.append(
                f"| oracle_current wall speedup | {mechanism.oracle_current_speedup:.3f}x "
                "| analysis-only upper bound, cannot satisfy the wall-clock gate |"
            )
        lines.append("")
        gap = mechanism.executed_work_ratio - mechanism.oracle_plan_work_ratio
        if gap > 0.05:
            lines.append(
                "The executor reclaims work when given good plans, and the deployed "
                "predictor does not supply them: the gap between the two ratios above is "
                f"{gap:.3f}. The remaining loss is scheduler quality, not executor "
                "structure."
            )
        else:
            lines.append(
                "The deployed scheduler is close to the oracle plan, so predictor quality "
                "is not the dominant remaining loss."
            )

    lines.extend(
        [
            "",
            "## Correctness And Diagnostics",
            "",
            (
                "Run 21070 GP leapfrog-count mismatches remain historical pre-repair "
                f"diagnostic evidence: {report.pre_repair_gp_mismatches} / "
                f"{report.pre_repair_gp_bucketed_rows} GP bucketed rows."
            ),
            "The repaired GP correctness gate is evaluated from T35 equivalence JSON.",
            "",
            "## Bucket-Count Scaling",
            "",
        ]
    )
    if report.bucket_loop_runtime_ratio is None:
        lines.append("Bucket-count scaling evidence was unavailable.")
    else:
        lines.append(
            "Repaired no-padding warm runtime max/min ratio across bucket-count rows: "
            f"{report.bucket_loop_runtime_ratio:.3f}."
        )

    lines.extend(["", "## Sources", ""])
    for criterion in report.criteria:
        lines.append(f"- {criterion.gate}: `{criterion.source}`")
    lines.append("")
    return "\n".join(lines)


def _latex_report(report: GateReport) -> str:
    lines = [
        "% Generated by: python -m abnuts.analysis.performance_gate",
        "\\section*{Performance Gate Before Paper Sweeps}",
        f"\\textbf{{Performance Gate: {_escape_latex(report.status)}}}",
        "",
        (
            "This report is generated from existing T32--T37R artifacts. "
            "It does not run a broad sweep."
        ),
        "",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{T37 performance gate criteria. A FAIL keeps broad sweeps blocked.}",
        "\\begin{tabular}{llll}",
        "\\hline",
        "Gate & Status & Value & Evidence \\\\",
        "\\hline",
    ]
    for criterion in report.criteria:
        lines.append(
            f"{_escape_latex(criterion.gate)} & "
            f"{_escape_latex(criterion.status)} & "
            f"{_escape_latex(criterion.value)} & "
            f"{_escape_latex(_shorten(criterion.evidence, 96))} \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
            "",
            "\\paragraph{Faster rows.}",
            (
                "Pre-repair run 21070 bucketed rows faster than monolithic: "
                f"{report.pre_repair_faster_rows}/{report.pre_repair_bucketed_rows}. "
                "These rows are negative diagnostic evidence from the pre-repair executor, "
                "not paper-scale speedup evidence."
            ),
            (
                "Repaired local non-analysis bucketed rows faster than monolithic: "
                f"{report.repaired_faster_rows}/{report.repaired_candidate_rows}."
            ),
            "",
        ]
    )
    if report.mechanism is not None:
        mechanism = report.mechanism
        lines.extend(
            [
                "\\paragraph{Mechanism versus scheduler.}",
                (
                    "Executed lane-steps are hardware independent and separate the executor "
                    "from the scheduler. The deployed scheduler reached an executed-work "
                    f"ratio of {mechanism.executed_work_ratio:.3f}, while oracle plans on the "
                    "same realized depths reach "
                    f"{mechanism.oracle_plan_work_ratio:.3f} through the same executor."
                ),
            ]
        )
        if mechanism.planner_share is not None:
            lines.append(
                f"Planning accounted for {mechanism.planner_share * 100:.1f}\\% of warm time."
            )
        if mechanism.oracle_current_speedup is not None:
            lines.append(
                "The oracle\\_current wall speedup of "
                f"{mechanism.oracle_current_speedup:.3f}x is an analysis-only upper bound "
                "and cannot satisfy the wall-clock gate."
            )
        lines.append("")
    lines.append("\\paragraph{Decision.}")
    if report.status == "PASS":
        lines.append("The gate passed; promote minimal HPC validation.")
    else:
        lines.append("The gate failed; broad sweeps remain blocked and T38 is not promoted.")
    lines.extend(["", f"Next task guidance: {_escape_latex(report.next_task_hint)}", ""])
    return "\n".join(lines)


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory already contains files: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"expected CSV at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _escape_latex(text: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def _shorten(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _jsonable_sources(sources: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sources.items():
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, (tuple, list)):
            result[key] = [str(item) for item in value]
        else:
            result[key] = value
    return result


def main(argv: list[str] | None = None) -> int:
    """Run the gate report command."""
    args = parse_args(argv)
    heterogeneous_summaries = tuple(args.heterogeneous_summary or DEFAULT_HETEROGENEOUS_SUMMARIES)
    negative_control_summaries = tuple(args.negative_control_summary or ())
    sources = {
        "pre_repair_run_dir": args.pre_repair_run_dir,
        "pre_repair_forensics_csv": args.pre_repair_forensics_csv,
        "correctness_dir": args.correctness_dir,
        "heterogeneous_summaries": heterogeneous_summaries,
        "oracle_gap_dir": args.oracle_gap_dir,
        "negative_control_summaries": negative_control_summaries,
    }
    try:
        report = build_gate_report(
            pre_repair_run_dir=args.pre_repair_run_dir,
            pre_repair_forensics_csv=args.pre_repair_forensics_csv,
            correctness_dir=args.correctness_dir,
            heterogeneous_summaries=heterogeneous_summaries,
            oracle_gap_dir=args.oracle_gap_dir,
            negative_control_summaries=negative_control_summaries,
            min_heterogeneous_speedup=args.min_heterogeneous_speedup,
            bucket_loop_max_runtime_ratio=args.bucket_loop_max_runtime_ratio,
            min_negative_control_speedup=args.min_negative_control_speedup,
            max_executed_work_ratio=args.max_executed_work_ratio,
        )
        summary_path, markdown_path, manifest_path, latex_path = write_outputs(
            report,
            out_dir=args.out,
            latex_out_dir=args.latex_out,
            sources=sources,
            overwrite=args.overwrite,
        )
    except (OSError, ValueError) as exc:
        print(f"performance_gate: {exc}", file=sys.stderr)
        return 2

    print(f"Performance gate result: {report.status}")
    print(f"Wrote gate CSV: {summary_path}")
    print(f"Wrote gate report: {markdown_path}")
    print(f"Wrote gate manifest: {manifest_path}")
    if latex_path is not None:
        print(f"Wrote gate LaTeX: {latex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
