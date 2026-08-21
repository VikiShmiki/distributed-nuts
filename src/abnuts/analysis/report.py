"""Generate LaTeX reports from raw ABNUTS benchmark outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from abnuts.analysis.aggregate import (
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
from abnuts.analysis.bundle import generate_aggregate_report
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
    write_result_macros,
    write_runtime_breakdown_table,
    write_sensitivity_coverage_table,
    write_sensitivity_summary,
    write_sensitivity_worst_group_table,
)
from abnuts.analysis.sensitivity import load_sensitivity

COMMAND = "python -m abnuts.analysis.report"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for LaTeX report generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw result directory.")
    parser.add_argument("--out", type=Path, required=True, help="LaTeX output directory.")
    parser.add_argument(
        "--include-sensitivity",
        action="store_true",
        help="Scan a raw result tree and generate sensitivity/worst-group tables.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "When scanning a raw result tree, emit TODO_RESULT markers for missing "
            "experiments instead of failing."
        ),
    )
    return parser.parse_args(argv)


def generate_report(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    include_sensitivity: bool = False,
    allow_missing: bool = False,
) -> tuple[tuple[Path, ...], tuple[Path, ...], tuple[Path, ...]]:
    """Generate LaTeX outputs for a raw benchmark summary."""
    input_path = Path(input_dir)
    if _is_raw_result_tree(input_path) and not include_sensitivity:
        result = generate_aggregate_report(
            input_path,
            out_dir,
            allow_missing=allow_missing,
            command=COMMAND,
        )
        return result.tables, result.figures, (*result.extras, *result.processed)

    if include_sensitivity:
        summary = load_sensitivity(input_dir)
        worst_table = write_sensitivity_worst_group_table(
            summary,
            out_dir,
            command=COMMAND,
        )
        coverage_table = write_sensitivity_coverage_table(
            summary,
            out_dir,
            command=COMMAND,
        )
        sensitivity_summary = write_sensitivity_summary(
            summary,
            out_dir,
            command=COMMAND,
        )
        return (worst_table, coverage_table), (), (sensitivity_summary,)
    if has_sbc_rank_histogram(input_dir):
        summary = load_sbc_rank_histogram(input_dir)
        figure = write_sbc_rank_histogram_figure(summary, out_dir, command=COMMAND)
        return (), (figure,), ()
    if has_trace_diagnostics(input_dir):
        summary = load_trace_diagnostics(input_dir)
        table = write_diagnostics_table(summary, out_dir, command=COMMAND)
        return (table,), (), ()
    if summary_has_oracle_gap(input_dir):
        summary = load_oracle_gap(input_dir)
        table = write_oracle_gap_table(summary, out_dir, command=COMMAND)
        interpretability = load_scheduler_interpretability(input_dir)
        figures = (
            write_oracle_gap_waterfall_figure(summary, out_dir, command=COMMAND),
            write_predictor_calibration_figure(
                interpretability.calibration,
                out_dir,
                command=COMMAND,
            ),
            write_padding_heatmap_figure(
                interpretability.padding,
                out_dir,
                command=COMMAND,
            ),
            write_speedup_vs_heterogeneity_figure(
                interpretability.heterogeneity,
                out_dir,
                command=COMMAND,
            ),
        )
        return (table,), figures, ()
    if summary_has_timing_breakdown(input_dir):
        summary = load_runtime_breakdown(input_dir)
        table = write_runtime_breakdown_table(summary, out_dir, command=COMMAND)
        figure = write_runtime_breakdown_figure(summary, out_dir, command=COMMAND)
        return (table,), (figure,), ()
    if summary_has_dimension_sweep(input_dir):
        summary = load_dimension_scaling(input_dir)
        table = write_dimension_scaling_table(summary, out_dir, command=COMMAND)
        figure = write_dimension_scaling_figure(summary, out_dir, command=COMMAND)
    elif summary_has_chain_sweep(input_dir):
        summary = load_chain_scaling(input_dir)
        table = write_chain_scaling_table(summary, out_dir, command=COMMAND)
        figure = write_chain_scaling_figure(summary, out_dir, command=COMMAND)
    else:
        summary = load_funnel_ablation(input_dir)
        table = write_bucket_ablation_table(summary, out_dir, command=COMMAND)
        figure = write_speedup_vs_bucket_size_figure(summary, out_dir, command=COMMAND)
    macros = write_result_macros(summary, out_dir, command=COMMAND)
    return (table,), (figure,), (macros,)


def main(argv: list[str] | None = None) -> int:
    """Run the report command."""
    args = parse_args(argv)
    try:
        tables, figures, extras = generate_report(
            args.input,
            args.out,
            include_sensitivity=args.include_sensitivity,
            allow_missing=args.allow_missing,
        )
    except (OSError, ValueError) as exc:
        print(
            f"Analysis report failed (input={args.input}, out={args.out}): {exc}",
            file=sys.stderr,
        )
        return 2

    for table in tables:
        print(f"Wrote LaTeX table: {table}")
    for figure in figures:
        print(f"Wrote LaTeX figure: {figure}")
    for extra in extras:
        if extra.name == "result_macros.tex":
            print(f"Wrote LaTeX macros: {extra}")
        elif extra.suffix == ".csv":
            print(f"Wrote processed CSV: {extra}")
        elif extra.name == "manifest.json":
            print(f"Wrote report manifest: {extra}")
        else:
            print(f"Wrote LaTeX summary: {extra}")
    return 0


def _is_raw_result_tree(input_dir: Path) -> bool:
    """Return true when ``input_dir`` looks like a multi-experiment raw root."""
    return (
        input_dir.is_dir()
        and not (input_dir / "summary.csv").exists()
        and not (input_dir / "diagnostics.csv").exists()
        and not (input_dir / "rank_histogram.csv").exists()
        and any(input_dir.rglob("summary.csv"))
    )


if __name__ == "__main__":
    raise SystemExit(main())
