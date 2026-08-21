"""Aggregate all available raw experiment outputs into one LaTeX bundle."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from abnuts.analysis.aggregate import (
    AblationSummary,
    ChainScalingSummary,
    DimensionScalingSummary,
    OracleGapSummary,
    has_sbc_rank_histogram,
    has_trace_diagnostics,
    load_chain_scaling,
    load_dimension_scaling,
    load_funnel_ablation,
    load_oracle_gap,
    load_runtime_breakdown,
    load_sbc_rank_histogram,
    load_scheduler_interpretability,
    load_trace_diagnostics,
    summary_has_chain_sweep,
    summary_has_dimension_sweep,
    summary_has_oracle_gap,
    summary_has_timing_breakdown,
)
from abnuts.analysis.latex_figures import (
    write_chain_scaling_figure,
    write_dimension_scaling_figure,
    write_oracle_gap_waterfall_figure,
    write_padding_heatmap_figure,
    write_predictor_calibration_figure,
    write_runtime_breakdown_figure,
    write_sbc_rank_histogram_figure,
    write_speedup_vs_bucket_size_figure,
    write_speedup_vs_heterogeneity_figure,
)
from abnuts.analysis.latex_tables import (
    write_bucket_ablation_table,
    write_chain_scaling_table,
    write_diagnostics_table,
    write_dimension_scaling_table,
    write_oracle_gap_table,
    write_runtime_breakdown_table,
    write_sensitivity_coverage_table,
    write_sensitivity_summary,
    write_sensitivity_worst_group_table,
)
from abnuts.analysis.sensitivity import load_sensitivity


@dataclass(frozen=True)
class AggregateReportResult:
    """Paths and warnings produced by the aggregate report command."""

    tables: tuple[Path, ...]
    figures: tuple[Path, ...]
    extras: tuple[Path, ...]
    processed: tuple[Path, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedStudy:
    """One generated study entry for the aggregate bundle summary."""

    key: str
    title: str
    source: Path
    outputs: tuple[Path, ...]
    note: str


@dataclass(frozen=True)
class MissingStudy:
    """One expected study that is missing from the raw result tree."""

    key: str
    title: str
    expected: str
    note: str


def generate_aggregate_report(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    allow_missing: bool,
    command: str,
) -> AggregateReportResult:
    """Generate the all-experiment LaTeX bundle from a raw result tree."""
    raw_root = Path(input_dir)
    latex_root = Path(out_dir)
    if not raw_root.exists():
        raise FileNotFoundError(f"raw result root does not exist: {raw_root}")
    latex_root.mkdir(parents=True, exist_ok=True)

    processed_root = _processed_dir_for(latex_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    tables: list[Path] = []
    figures: list[Path] = []
    extras: list[Path] = []
    warnings: list[str] = []
    generated: list[GeneratedStudy] = []
    missing: list[MissingStudy] = []
    summaries: dict[str, object] = {}

    funnel_dir = _find_summary_dir(
        raw_root,
        preferred=("funnel_ablation/tiny", "hvp_tiny"),
        predicate=_is_standard_ablation_summary,
    )
    if funnel_dir is None:
        missing.append(
            MissingStudy(
                key="funnel_ablation",
                title="Funnel bucket ablation",
                expected="results/raw/funnel_ablation/<run>/summary.csv",
                note="TODO_RESULT: run funnel ablation before making paper-scale claims.",
            )
        )
    else:
        summary = load_funnel_ablation(funnel_dir)
        table = write_bucket_ablation_table(summary, latex_root, command=command)
        figure = write_speedup_vs_bucket_size_figure(summary, latex_root, command=command)
        tables.append(table)
        figures.append(figure)
        generated.append(
            GeneratedStudy(
                key="funnel_ablation",
                title="Funnel bucket ablation",
                source=summary.source_csv,
                outputs=(table, figure),
                note=_evidence_note(summary.source_csv),
            )
        )
        summaries["funnel_ablation"] = summary

    chain_dir = _find_summary_dir(
        raw_root,
        preferred=("chain_scaling/tiny",),
        predicate=summary_has_chain_sweep,
    )
    if chain_dir is None:
        missing.append(
            MissingStudy(
                key="chain_scaling",
                title="Chain scaling",
                expected="results/raw/chain_scaling/<run>/summary.csv",
                note="TODO_RESULT: run chain-scaling sweep for paper-scale evidence.",
            )
        )
    else:
        summary = load_chain_scaling(chain_dir)
        table = write_chain_scaling_table(summary, latex_root, command=command)
        figure = write_chain_scaling_figure(summary, latex_root, command=command)
        tables.append(table)
        figures.append(figure)
        generated.append(
            GeneratedStudy(
                key="chain_scaling",
                title="Chain scaling",
                source=summary.source_csv,
                outputs=(table, figure),
                note=_evidence_note(summary.source_csv),
            )
        )
        summaries["chain_scaling"] = summary

    dimension_dir = _find_summary_dir(
        raw_root,
        preferred=("dimension_scaling/tiny",),
        predicate=summary_has_dimension_sweep,
    )
    if dimension_dir is None:
        missing.append(
            MissingStudy(
                key="dimension_scaling",
                title="Dimension scaling",
                expected="results/raw/dimension_scaling/<run>/summary.csv",
                note="TODO_RESULT: run dimension-scaling sweep for paper-scale evidence.",
            )
        )
    else:
        summary = load_dimension_scaling(dimension_dir)
        table = write_dimension_scaling_table(summary, latex_root, command=command)
        figure = write_dimension_scaling_figure(summary, latex_root, command=command)
        tables.append(table)
        figures.append(figure)
        generated.append(
            GeneratedStudy(
                key="dimension_scaling",
                title="Dimension scaling",
                source=summary.source_csv,
                outputs=(table, figure),
                note=_evidence_note(summary.source_csv),
            )
        )
        summaries["dimension_scaling"] = summary

    oracle_dir = _find_summary_dir(
        raw_root,
        preferred=("oracle_gap/tiny",),
        predicate=summary_has_oracle_gap,
    )
    if oracle_dir is None:
        missing.append(
            MissingStudy(
                key="oracle_gap",
                title="Oracle-gap decomposition",
                expected="results/raw/oracle_gap/<run>/summary.csv",
                note="TODO_RESULT: run oracle-gap decomposition; oracle-current must stay labeled.",
            )
        )
    else:
        summary = load_oracle_gap(oracle_dir)
        interpretability = load_scheduler_interpretability(oracle_dir)
        table = write_oracle_gap_table(summary, latex_root, command=command)
        generated_figures = (
            write_oracle_gap_waterfall_figure(summary, latex_root, command=command),
            write_predictor_calibration_figure(
                interpretability.calibration,
                latex_root,
                command=command,
            ),
            write_padding_heatmap_figure(
                interpretability.padding,
                latex_root,
                command=command,
            ),
            write_speedup_vs_heterogeneity_figure(
                interpretability.heterogeneity,
                latex_root,
                command=command,
            ),
        )
        tables.append(table)
        figures.extend(generated_figures)
        generated.append(
            GeneratedStudy(
                key="oracle_gap",
                title="Oracle-gap decomposition",
                source=summary.source_csv,
                outputs=(table, *generated_figures),
                note=(
                    _evidence_note(summary.source_csv)
                    + " Oracle-current rows are analysis-only upper bounds."
                ),
            )
        )
        summaries["oracle_gap"] = summary

    diagnostics_dir = _find_dir(
        raw_root,
        preferred=("long_trace/tiny", "trace_qa/tiny"),
        predicate=has_trace_diagnostics,
    )
    if diagnostics_dir is None:
        missing.append(
            MissingStudy(
                key="diagnostics",
                title="Long-trace diagnostics",
                expected="results/raw/long_trace/<run>/diagnostics.csv",
                note="TODO_RESULT: run long-trace correctness diagnostics.",
            )
        )
    else:
        summary = load_trace_diagnostics(diagnostics_dir)
        table = write_diagnostics_table(summary, latex_root, command=command)
        tables.append(table)
        generated.append(
            GeneratedStudy(
                key="diagnostics",
                title="Long-trace diagnostics",
                source=summary.source_csv,
                outputs=(table,),
                note=_evidence_note(summary.source_csv),
            )
        )

    sbc_dir = _find_dir(
        raw_root,
        preferred=("sbc/tiny",),
        predicate=has_sbc_rank_histogram,
    )
    if sbc_dir is None:
        missing.append(
            MissingStudy(
                key="sbc",
                title="Simulation-based calibration",
                expected="results/raw/sbc/<run>/rank_histogram.csv",
                note="TODO_RESULT: run SBC before making calibration claims.",
            )
        )
    else:
        summary = load_sbc_rank_histogram(sbc_dir)
        figure = write_sbc_rank_histogram_figure(summary, latex_root, command=command)
        figures.append(figure)
        generated.append(
            GeneratedStudy(
                key="sbc",
                title="Simulation-based calibration",
                source=summary.source_csv,
                outputs=(figure,),
                note=_evidence_note(summary.source_csv),
            )
        )

    runtime_dir = _find_summary_dir(
        raw_root,
        preferred=("profiling/tiny",),
        predicate=summary_has_timing_breakdown,
    )
    if runtime_dir is None:
        missing.append(
            MissingStudy(
                key="profiling",
                title="Runtime breakdown",
                expected="results/raw/profiling/<run>/summary.csv",
                note="TODO_RESULT: run timing-breakdown profiling before paper claims.",
            )
        )
    else:
        summary = load_runtime_breakdown(runtime_dir)
        table = write_runtime_breakdown_table(summary, latex_root, command=command)
        figure = write_runtime_breakdown_figure(summary, latex_root, command=command)
        tables.append(table)
        figures.append(figure)
        generated.append(
            GeneratedStudy(
                key="profiling",
                title="Runtime breakdown",
                source=summary.source_csv,
                outputs=(table, figure),
                note=_evidence_note(summary.source_csv),
            )
        )
        summaries["profiling"] = summary

    try:
        sensitivity = load_sensitivity(raw_root)
    except (OSError, ValueError) as exc:
        missing.append(
            MissingStudy(
                key="sensitivity",
                title="Sensitivity and worst-group reporting",
                expected="bucketed speedup rows under results/raw/**/summary.csv",
                note=f"TODO_RESULT: sensitivity report skipped ({exc}).",
            )
        )
    else:
        worst_table = write_sensitivity_worst_group_table(
            sensitivity,
            latex_root,
            command=command,
        )
        coverage_table = write_sensitivity_coverage_table(
            sensitivity,
            latex_root,
            command=command,
        )
        sensitivity_tex = write_sensitivity_summary(
            sensitivity,
            latex_root,
            command=command,
        )
        tables.extend((worst_table, coverage_table))
        extras.append(sensitivity_tex)
        generated.append(
            GeneratedStudy(
                key="sensitivity",
                title="Sensitivity and worst-group reporting",
                source=raw_root,
                outputs=(worst_table, coverage_table, sensitivity_tex),
                note=sensitivity.evidence_note,
            )
        )

    paper_scale_missing = _paper_scale_missing(generated)
    missing.extend(paper_scale_missing)

    if missing and not allow_missing:
        detail = "; ".join(f"{item.key}: {item.expected}" for item in missing)
        raise ValueError(f"missing required aggregate report inputs ({detail})")

    processed = list(_write_processed_csvs(raw_root, processed_root, missing))
    macros = _write_bundle_macros(
        summaries=summaries,
        raw_root=raw_root,
        out_dir=latex_root,
        command=command,
    )
    extras.append(macros)
    summary_tex = _write_results_summary(
        raw_root=raw_root,
        out_dir=latex_root,
        processed_root=processed_root,
        command=command,
        generated=generated,
        missing=missing,
        warnings=warnings,
    )
    extras.append(summary_tex)
    manifest = _write_manifest(
        raw_root=raw_root,
        out_dir=latex_root,
        processed_root=processed_root,
        generated=generated,
        missing=missing,
        tables=tables,
        figures=figures,
        extras=extras,
        processed=processed,
        warnings=warnings,
        command=command,
        allow_missing=allow_missing,
    )
    extras.append(manifest)

    return AggregateReportResult(
        tables=tuple(tables),
        figures=tuple(figures),
        extras=tuple(extras),
        processed=tuple(processed),
        warnings=tuple(warnings),
    )


def _find_summary_dir(
    raw_root: Path,
    *,
    preferred: tuple[str, ...],
    predicate: Any,
) -> Path | None:
    return _find_dir(
        raw_root,
        preferred=preferred,
        predicate=lambda path: (path / "summary.csv").exists() and _safe_predicate(
            predicate,
            path,
        ),
    )


def _find_dir(
    raw_root: Path,
    *,
    preferred: tuple[str, ...],
    predicate: Any,
) -> Path | None:
    for rel_path in preferred:
        candidate = raw_root / rel_path
        if candidate.exists() and _safe_predicate(predicate, candidate):
            return candidate
    for candidate in sorted({path.parent for path in raw_root.rglob("*")}):
        if _safe_predicate(predicate, candidate):
            return candidate
    return None


def _safe_predicate(predicate: Any, path: Path) -> bool:
    try:
        return bool(predicate(path))
    except (OSError, ValueError, KeyError):
        return False


def _is_standard_ablation_summary(path: Path) -> bool:
    return (
        (path / "summary.csv").exists()
        and not summary_has_oracle_gap(path)
        and not summary_has_timing_breakdown(path)
        and not summary_has_chain_sweep(path)
        and not summary_has_dimension_sweep(path)
    )


def _write_processed_csvs(
    raw_root: Path,
    processed_root: Path,
    missing: list[MissingStudy],
) -> tuple[Path, ...]:
    processed: list[Path] = []
    processed.append(
        _write_collated_csv(
            sorted(raw_root.rglob("summary.csv")),
            processed_root / "all_summary.csv",
        )
    )
    diagnostics = sorted(raw_root.rglob("diagnostics.csv"))
    if diagnostics:
        processed.append(
            _write_collated_csv(diagnostics, processed_root / "all_diagnostics.csv")
        )
    rank_histograms = sorted(raw_root.rglob("rank_histogram.csv"))
    if rank_histograms:
        processed.append(
            _write_collated_csv(
                rank_histograms,
                processed_root / "all_rank_histogram.csv",
            )
        )
    processed.append(_write_equivalence_checks(raw_root, processed_root))
    processed.append(_write_missing_results_csv(missing, processed_root))
    return tuple(processed)


def _write_collated_csv(source_paths: list[Path], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    fieldnames: list[str] = ["source_csv"]
    for source_path in source_paths:
        with source_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            source_rows = list(reader)
        for name in reader.fieldnames or ():
            if name not in fieldnames:
                fieldnames.append(name)
        for row in source_rows:
            rows.append({"source_csv": str(source_path), **row})

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return out_path


def _write_equivalence_checks(raw_root: Path, processed_root: Path) -> Path:
    out_path = processed_root / "equivalence_checks.csv"
    fieldnames = [
        "source_json",
        "check_kind",
        "max_position_delta",
        "positions_equal",
        "saved_positions_equal",
        "diagnostics_equal",
        "final_rng_keys_equal",
    ]
    rows: list[dict[str, str]] = []
    for source_path in sorted(
        [
            *raw_root.rglob("equivalence.json"),
            *raw_root.rglob("trace_qa.json"),
        ]
    ):
        with source_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rows.append(
            {
                "source_json": str(source_path),
                "check_kind": source_path.stem,
                "max_position_delta": str(payload.get("max_position_delta", "")),
                "positions_equal": str(payload.get("positions_equal", "")),
                "saved_positions_equal": str(payload.get("saved_positions_equal", "")),
                "diagnostics_equal": str(payload.get("diagnostics_equal", "")),
                "final_rng_keys_equal": str(payload.get("final_rng_keys_equal", "")),
            }
        )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def _write_missing_results_csv(
    missing: list[MissingStudy],
    processed_root: Path,
) -> Path:
    out_path = processed_root / "missing_results.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["key", "title", "expected", "note"],
            lineterminator="\n",
        )
        writer.writeheader()
        for item in missing:
            writer.writerow(
                {
                    "key": item.key,
                    "title": item.title,
                    "expected": item.expected,
                    "note": item.note,
                }
            )
    return out_path


def _write_bundle_macros(
    *,
    summaries: dict[str, object],
    raw_root: Path,
    out_dir: Path,
    command: str,
) -> Path:
    macros_dir = out_dir / "macros"
    macros_dir.mkdir(parents=True, exist_ok=True)
    macros_path = macros_dir / "result_macros.tex"
    speedup_candidates = _speedup_candidates(summaries)
    best_speedup, best_config = max(speedup_candidates, default=(None, "TODO_RESULT"))
    one_step_delta = _min_delta(raw_root.rglob("equivalence.json"))
    trace_delta = _min_delta(raw_root.rglob("trace_qa.json"))
    lines = [
        f"% Generated by: {command}",
        f"% Source result root: {raw_root}",
        "% TODO_RESULT values indicate missing paper-scale or unavailable raw evidence.",
        f"\\newcommand{{\\BestWarmSpeedup}}{{{_macro_value(best_speedup)}}}",
        f"\\newcommand{{\\BestWarmSpeedupConfig}}{{{_escape_latex(best_config)}}}",
        "\\newcommand{\\OneStepMaxDelta}"
        f"{{{_macro_value(one_step_delta, digits=6)}}}",
        "\\newcommand{\\TraceQAMaxDelta}"
        f"{{{_macro_value(trace_delta, digits=6)}}}",
    ]
    if isinstance(summaries.get("funnel_ablation"), AblationSummary):
        summary = summaries["funnel_ablation"]
        history = [
            row.mean_speedup
            for row in summary.rows
            if row.predictor == "history"
        ]
        lines.append(
            "\\newcommand{\\FunnelHistorySpeedupCtiny}"
            f"{{{_macro_value(max(history) if history else None)}}}"
        )
    else:
        lines.append("\\newcommand{\\FunnelHistorySpeedupCtiny}{TODO_RESULT}")
    if isinstance(summaries.get("chain_scaling"), ChainScalingSummary):
        summary = summaries["chain_scaling"]
        lines.append(
            f"\\newcommand{{\\ChainScalingBestChains}}{{{summary.best_row.num_chains}}}"
        )
        lines.append(
            "\\newcommand{\\ChainScalingBestSpeedup}"
            f"{{{summary.best_row.mean_speedup:.3f}}}"
        )
    else:
        lines.append("\\newcommand{\\ChainScalingBestChains}{TODO_RESULT}")
        lines.append("\\newcommand{\\ChainScalingBestSpeedup}{TODO_RESULT}")
    if isinstance(summaries.get("dimension_scaling"), DimensionScalingSummary):
        summary = summaries["dimension_scaling"]
        lines.append(
            "\\newcommand{\\DimensionScalingBestDimension}"
            f"{{{summary.best_row.dimension}}}"
        )
        lines.append(
            "\\newcommand{\\DimensionScalingBestSpeedup}"
            f"{{{summary.best_row.mean_speedup:.3f}}}"
        )
    else:
        lines.append("\\newcommand{\\DimensionScalingBestDimension}{TODO_RESULT}")
        lines.append("\\newcommand{\\DimensionScalingBestSpeedup}{TODO_RESULT}")
    if isinstance(summaries.get("oracle_gap"), OracleGapSummary):
        deployable = [
            row.mean_speedup
            for row in summaries["oracle_gap"].rows
            if not row.is_analysis_upper_bound
        ]
        upper = [
            row.mean_speedup
            for row in summaries["oracle_gap"].rows
            if row.is_analysis_upper_bound
        ]
        lines.append(
            "\\newcommand{\\OracleGapBestDeployableSpeedup}"
            f"{{{_macro_value(max(deployable) if deployable else None)}}}"
        )
        lines.append(
            "\\newcommand{\\OracleCurrentUpperBoundSpeedup}"
            f"{{{_macro_value(max(upper) if upper else None)}}}"
        )
    else:
        lines.append("\\newcommand{\\OracleGapBestDeployableSpeedup}{TODO_RESULT}")
        lines.append("\\newcommand{\\OracleCurrentUpperBoundSpeedup}{TODO_RESULT}")
    lines.append("\\newcommand{\\PaperScaleEvidenceStatus}{TODO_RESULT}")
    lines.append("")
    macros_path.write_text("\n".join(lines), encoding="utf-8")
    return macros_path


def _write_results_summary(
    *,
    raw_root: Path,
    out_dir: Path,
    processed_root: Path,
    command: str,
    generated: list[GeneratedStudy],
    missing: list[MissingStudy],
    warnings: list[str],
) -> Path:
    out_path = out_dir / "results_summary.tex"
    lines = [
        f"% Generated by: {command}",
        f"% Source result root: {raw_root}",
        f"% Processed CSV root: {processed_root}",
        "\\paragraph{Aggregate result bundle.}",
        (
            "This bundle was generated from available raw outputs. "
            "Tiny/smoke outputs are included for pipeline validation; paper-scale "
            "claims remain marked as TODO\\_RESULT until full runs complete."
        ),
        "",
        "\\input{macros/result_macros}",
        "",
        "\\paragraph{Generated artifacts.}",
    ]
    for study in generated:
        lines.append(
            f"% {study.key}: source={study.source}; note={study.note}"
        )
        lines.append(f"\\paragraph{{{_escape_latex(study.title)}.}}")
        lines.append(_escape_latex(study.note))
        for output in study.outputs:
            if output.suffix == ".tex":
                rel = output.relative_to(out_dir).with_suffix("")
                lines.append(f"\\input{{{rel.as_posix()}}}")
        lines.append("")

    if missing:
        lines.extend(["\\paragraph{Missing or non-paper-scale evidence.}"])
        for item in missing:
            lines.append(
                f"TODO_RESULT: {_escape_latex(item.title)} -- "
                f"{_escape_latex(item.note)} Expected: {_escape_latex(item.expected)}."
            )
        lines.append("")

    if warnings:
        lines.extend(["\\paragraph{Warnings.}"])
        for warning in warnings:
            lines.append(f"TODO_RESULT: {_escape_latex(warning)}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def _write_manifest(
    *,
    raw_root: Path,
    out_dir: Path,
    processed_root: Path,
    generated: list[GeneratedStudy],
    missing: list[MissingStudy],
    tables: list[Path],
    figures: list[Path],
    extras: list[Path],
    processed: list[Path],
    warnings: list[str],
    command: str,
    allow_missing: bool,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    latex_files = sorted(
        {
            str(path.relative_to(out_dir))
            for path in (*tables, *figures, *extras)
            if path.suffix == ".tex"
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "input_root": str(raw_root),
        "output_root": str(out_dir),
        "processed_root": str(processed_root),
        "allow_missing": allow_missing,
        "latex_files": latex_files,
        "processed_files": [str(path) for path in processed],
        "generated_studies": [
            {
                "key": study.key,
                "title": study.title,
                "source": str(study.source),
                "outputs": [str(path.relative_to(out_dir)) for path in study.outputs],
                "note": study.note,
            }
            for study in generated
        ],
        "missing_results": [
            {
                "key": item.key,
                "title": item.title,
                "expected": item.expected,
                "note": item.note,
            }
            for item in missing
        ],
        "warnings": warnings,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _processed_dir_for(latex_root: Path) -> Path:
    parts = latex_root.parts
    if "latex" not in parts:
        return latex_root / "processed"
    index = parts.index("latex")
    return Path(*parts[:index], "processed", *parts[index + 1 :])


def _paper_scale_missing(generated: list[GeneratedStudy]) -> list[MissingStudy]:
    missing: list[MissingStudy] = []
    for study in generated:
        if _is_tiny_path(study.source):
            missing.append(
                MissingStudy(
                    key=f"{study.key}_paper_scale",
                    title=f"{study.title} paper-scale run",
                    expected=str(study.source).replace("/tiny/", "/<paper-run>/"),
                    note=(
                        "TODO_RESULT: only tiny/smoke evidence is available in this "
                        "aggregate bundle."
                    ),
                )
            )
    return missing


def _speedup_candidates(summaries: dict[str, object]) -> list[tuple[float, str]]:
    candidates: list[tuple[float, str]] = []
    if isinstance(summaries.get("funnel_ablation"), AblationSummary):
        row = summaries["funnel_ablation"].best_row
        candidates.append(
            (
                row.mean_speedup,
                f"funnel ablation, {row.predictor}, S={row.bucket_size}",
            )
        )
    if isinstance(summaries.get("chain_scaling"), ChainScalingSummary):
        row = summaries["chain_scaling"].best_row
        candidates.append(
            (
                row.mean_speedup,
                f"chain scaling, C={row.num_chains}, {row.predictor}, S={row.bucket_size}",
            )
        )
    if isinstance(summaries.get("dimension_scaling"), DimensionScalingSummary):
        row = summaries["dimension_scaling"].best_row
        candidates.append(
            (
                row.mean_speedup,
                f"dimension scaling, D={row.dimension}, {row.predictor}, S={row.bucket_size}",
            )
        )
    if isinstance(summaries.get("oracle_gap"), OracleGapSummary):
        deployable = [
            row
            for row in summaries["oracle_gap"].rows
            if not row.is_analysis_upper_bound
        ]
        if deployable:
            row = max(deployable, key=lambda value: value.mean_speedup)
            candidates.append(
                (
                    row.mean_speedup,
                    f"oracle gap, {row.scheduler_label}, S={row.bucket_size}",
                )
            )
    return candidates


def _min_delta(paths: Any) -> float | None:
    values: list[float] = []
    for path in sorted(paths):
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        value = payload.get("max_position_delta")
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return max(values)


def _macro_value(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "TODO_RESULT"
    return f"{value:.{digits}f}"


def _evidence_note(source: Path) -> str:
    if _is_tiny_path(source):
        return "Smoke-only tiny result; not paper-scale evidence."
    return "Available result; verify uncertainty before claiming paper-scale evidence."


def _is_tiny_path(source: Path) -> bool:
    text = source.as_posix()
    return "/tiny" in text or text.endswith("_tiny/summary.csv")


def _escape_latex(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )
