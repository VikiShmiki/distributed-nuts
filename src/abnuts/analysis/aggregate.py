"""Aggregate raw benchmark CSV outputs for LaTeX reporting."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from abnuts.analysis.diagnostics import TraceDiagnosticRow, load_diagnostics_csv


@dataclass(frozen=True)
class AblationRow:
    """One aggregated funnel-ablation result row."""

    predictor: str
    bucket_size: int
    runs: int
    mean_t_mono: float
    mean_t_bucket: float
    mean_speedup: float
    mean_padding_ratio: float
    total_divergences: int
    max_tree_depth: int


@dataclass(frozen=True)
class AblationSummary:
    """Aggregated funnel-ablation data and source metadata."""

    source_csv: Path
    rows: tuple[AblationRow, ...]

    @property
    def best_row(self) -> AblationRow:
        """Return the row with the largest mean speedup."""
        if not self.rows:
            raise ValueError("cannot choose a best row from an empty ablation summary")
        return max(self.rows, key=lambda row: (row.mean_speedup, -row.bucket_size, row.predictor))


@dataclass(frozen=True)
class ChainScalingRow:
    """One aggregated chain-scaling result row."""

    num_chains: int
    predictor: str
    bucket_size: int
    runs: int
    mean_t_mono: float
    mean_t_bucket: float
    mean_speedup: float
    mean_padding_ratio: float


@dataclass(frozen=True)
class ChainScalingSummary:
    """Aggregated chain-scaling data and source metadata."""

    source_csv: Path
    rows: tuple[ChainScalingRow, ...]

    @property
    def best_row(self) -> ChainScalingRow:
        """Return the row with the largest mean speedup."""
        if not self.rows:
            raise ValueError("cannot choose a best row from an empty chain-scaling summary")
        return max(self.rows, key=lambda row: (row.mean_speedup, row.num_chains))


@dataclass(frozen=True)
class DimensionScalingRow:
    """One aggregated dimension-scaling result row."""

    dimension: int
    predictor: str
    bucket_size: int
    runs: int
    mean_t_mono: float
    mean_t_bucket: float
    mean_speedup: float
    mean_padding_ratio: float


@dataclass(frozen=True)
class DimensionScalingSummary:
    """Aggregated dimension-scaling data and source metadata."""

    source_csv: Path
    rows: tuple[DimensionScalingRow, ...]

    @property
    def best_row(self) -> DimensionScalingRow:
        """Return the row with the largest mean speedup."""
        if not self.rows:
            raise ValueError("cannot choose a best row from an empty dimension-scaling summary")
        return max(self.rows, key=lambda row: (row.mean_speedup, row.dimension))


@dataclass(frozen=True)
class DiagnosticsSummary:
    """Trace-level diagnostics data and source metadata."""

    source_csv: Path
    rows: tuple[TraceDiagnosticRow, ...]


@dataclass(frozen=True)
class OracleGapRow:
    """One aggregated oracle-gap scheduler result row."""

    scheduler_mode: str
    scheduler_label: str
    bucket_size: int
    is_analysis_upper_bound: bool
    runs: int
    mean_t_mode: float
    mean_speedup: float
    mean_padding_ratio: float
    mean_predictor_abs_error: float
    mean_sum_bucket_realized_max: float
    mean_global_realized_max: float
    mean_bucket_max_over_global_max: float
    mean_cold_compile_seconds: float = 0.0
    mean_warm_iteration_seconds: float = 0.0
    mean_planner_seconds: float = 0.0
    mean_executor_seconds: float = 0.0
    mean_gather_seconds: float = 0.0
    mean_scatter_seconds: float = 0.0
    mean_gather_scatter_seconds: float = 0.0
    mean_hvp_seconds: float = 0.0
    mean_unattributed_seconds: float = 0.0
    mean_non_executor_overhead_seconds: float = 0.0
    dominant_warm_component: str = "unavailable"
    dominant_warm_component_seconds: float = 0.0


@dataclass(frozen=True)
class OracleGapSummary:
    """Aggregated oracle-gap data and source metadata."""

    source_csv: Path
    rows: tuple[OracleGapRow, ...]


@dataclass(frozen=True)
class PredictorCalibrationRow:
    """One predictor calibration point for scheduler interpretability."""

    scheduler_mode: str
    scheduler_label: str
    bucket_size: int
    is_analysis_upper_bound: bool
    predicted_work: float
    realized_work: float
    abs_error: float
    is_summary_proxy: bool


@dataclass(frozen=True)
class PredictorCalibrationSummary:
    """Predictor calibration data and source metadata."""

    source_csv: Path
    rows: tuple[PredictorCalibrationRow, ...]


@dataclass(frozen=True)
class PaddingHeatmapRow:
    """One bucket occupancy row for padding heatmap figures."""

    scheduler_mode: str
    scheduler_label: str
    bucket_size: int
    is_analysis_upper_bound: bool
    step: int
    bucket_index: int
    padding_count: int
    fill_fraction: float
    bucket_realized_max: float


@dataclass(frozen=True)
class PaddingHeatmapSummary:
    """Padding heatmap data and source metadata."""

    source_csv: Path
    rows: tuple[PaddingHeatmapRow, ...]


@dataclass(frozen=True)
class SpeedupHeterogeneityRow:
    """One speedup-versus-heterogeneity point."""

    scheduler_mode: str
    scheduler_label: str
    bucket_size: int
    is_analysis_upper_bound: bool
    speedup: float
    heterogeneity: float


@dataclass(frozen=True)
class SpeedupHeterogeneitySummary:
    """Speedup-versus-heterogeneity data and source metadata."""

    source_csv: Path
    rows: tuple[SpeedupHeterogeneityRow, ...]


@dataclass(frozen=True)
class SBCRankHistogramRow:
    """One rank-bin count for an SBC rank histogram."""

    method: str
    rank: int
    count: int
    num_replicates: int
    num_posterior_samples: int
    is_smoke_only: bool
    evidence_note: str


@dataclass(frozen=True)
class SBCRankHistogramSummary:
    """SBC rank histogram data and source metadata."""

    source_csv: Path
    rows: tuple[SBCRankHistogramRow, ...]

    @property
    def is_smoke_only(self) -> bool:
        """Return true when any histogram row is marked smoke-only."""
        return any(row.is_smoke_only for row in self.rows)


@dataclass(frozen=True)
class RuntimeBreakdownRow:
    """One aggregated runtime component row for profiling reports."""

    method: str
    predictor: str
    bucket_size: int
    runs: int
    mean_cold_run_seconds: float
    mean_cold_compile_seconds: float
    mean_warm_iteration_seconds: float
    mean_planner_seconds: float
    mean_executor_seconds: float
    mean_gather_seconds: float
    mean_scatter_seconds: float
    mean_hvp_seconds: float
    mean_unattributed_seconds: float
    mean_profiler_marker_count: float


@dataclass(frozen=True)
class RuntimeBreakdownSummary:
    """Aggregated runtime breakdown data and source metadata."""

    source_csv: Path
    rows: tuple[RuntimeBreakdownRow, ...]


@dataclass(frozen=True)
class SchedulerInterpretabilitySummary:
    """All data needed for scheduler interpretability figures."""

    calibration: PredictorCalibrationSummary
    padding: PaddingHeatmapSummary
    heterogeneity: SpeedupHeterogeneitySummary


def load_funnel_ablation(input_dir: str | Path) -> AblationSummary:
    """Load and aggregate a raw funnel-ablation ``summary.csv``."""
    source_csv = Path(input_dir) / "summary.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"expected benchmark summary CSV at {source_csv}")

    raw_rows = _read_summary_rows(source_csv)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in raw_rows:
        key = (row["predictor"], int(row["bucket_size"]))
        grouped.setdefault(key, []).append(row)

    rows = tuple(
        _aggregate_group(predictor, bucket_size, group)
        for (predictor, bucket_size), group in sorted(grouped.items(), key=_sort_key)
    )
    return AblationSummary(source_csv=source_csv, rows=rows)


def load_chain_scaling(input_dir: str | Path) -> ChainScalingSummary:
    """Load and aggregate a raw chain-scaling ``summary.csv``."""
    source_csv = Path(input_dir) / "summary.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"expected benchmark summary CSV at {source_csv}")

    raw_rows = _read_summary_rows(source_csv)
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = {}
    for row in raw_rows:
        key = (int(row["num_chains"]), row["predictor"], int(row["bucket_size"]))
        grouped.setdefault(key, []).append(row)

    rows = tuple(
        _aggregate_chain_group(num_chains, predictor, bucket_size, group)
        for (num_chains, predictor, bucket_size), group in sorted(grouped.items())
    )
    return ChainScalingSummary(source_csv=source_csv, rows=rows)


def load_dimension_scaling(input_dir: str | Path) -> DimensionScalingSummary:
    """Load and aggregate a raw dimension-scaling ``summary.csv``."""
    source_csv = Path(input_dir) / "summary.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"expected benchmark summary CSV at {source_csv}")

    raw_rows = _read_summary_rows(source_csv)
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = {}
    for row in raw_rows:
        key = (int(row["dimension"]), row["predictor"], int(row["bucket_size"]))
        grouped.setdefault(key, []).append(row)

    rows = tuple(
        _aggregate_dimension_group(dimension, predictor, bucket_size, group)
        for (dimension, predictor, bucket_size), group in sorted(grouped.items())
    )
    return DimensionScalingSummary(source_csv=source_csv, rows=rows)


def load_trace_diagnostics(input_dir: str | Path) -> DiagnosticsSummary:
    """Load trace diagnostics from a correctness QA run."""
    source_csv = Path(input_dir) / "diagnostics.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"expected diagnostics CSV at {source_csv}")
    return DiagnosticsSummary(source_csv=source_csv, rows=load_diagnostics_csv(input_dir))


def load_oracle_gap(input_dir: str | Path) -> OracleGapSummary:
    """Load and aggregate a raw oracle-gap ``summary.csv``."""
    source_csv = Path(input_dir) / "summary.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"expected oracle-gap summary CSV at {source_csv}")

    raw_rows = _read_oracle_gap_rows(source_csv)
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in raw_rows:
        key = (row["scheduler_mode"], int(row["bucket_size"]))
        grouped.setdefault(key, []).append(row)

    rows = tuple(
        _aggregate_oracle_gap_group(scheduler_mode, bucket_size, group)
        for (scheduler_mode, bucket_size), group in sorted(
            grouped.items(),
            key=_oracle_gap_sort_key,
        )
    )
    return OracleGapSummary(source_csv=source_csv, rows=rows)


def load_scheduler_interpretability(input_dir: str | Path) -> SchedulerInterpretabilitySummary:
    """Load detailed scheduler interpretability data, falling back to summaries."""
    return SchedulerInterpretabilitySummary(
        calibration=load_predictor_calibration(input_dir),
        padding=load_padding_heatmap(input_dir),
        heterogeneity=load_speedup_heterogeneity(input_dir),
    )


def load_predictor_calibration(input_dir: str | Path) -> PredictorCalibrationSummary:
    """Load predicted-vs-realized work rows for scheduler calibration figures."""
    source_csv = Path(input_dir) / "predictor_calibration.csv"
    if source_csv.exists():
        rows = tuple(_read_predictor_calibration_rows(source_csv))
        return PredictorCalibrationSummary(source_csv=source_csv, rows=rows)

    oracle_summary = load_oracle_gap(input_dir)
    rows = tuple(_fallback_predictor_calibration_rows(oracle_summary))
    return PredictorCalibrationSummary(source_csv=oracle_summary.source_csv, rows=rows)


def load_padding_heatmap(input_dir: str | Path) -> PaddingHeatmapSummary:
    """Load per-bucket padding rows for heatmap figures."""
    source_csv = Path(input_dir) / "padding_heatmap.csv"
    if source_csv.exists():
        rows = tuple(_read_padding_heatmap_rows(source_csv))
        return PaddingHeatmapSummary(source_csv=source_csv, rows=rows)

    oracle_summary = load_oracle_gap(input_dir)
    rows = tuple(_fallback_padding_heatmap_rows(oracle_summary))
    return PaddingHeatmapSummary(source_csv=oracle_summary.source_csv, rows=rows)


def load_speedup_heterogeneity(input_dir: str | Path) -> SpeedupHeterogeneitySummary:
    """Load speedup and heterogeneity rows for scheduler interpretability."""
    source_csv = Path(input_dir) / "speedup_heterogeneity.csv"
    if source_csv.exists():
        rows = tuple(_read_speedup_heterogeneity_rows(source_csv))
        return SpeedupHeterogeneitySummary(source_csv=source_csv, rows=rows)

    oracle_summary = load_oracle_gap(input_dir)
    rows = tuple(_fallback_speedup_heterogeneity_rows(oracle_summary))
    return SpeedupHeterogeneitySummary(source_csv=oracle_summary.source_csv, rows=rows)


def load_sbc_rank_histogram(input_dir: str | Path) -> SBCRankHistogramSummary:
    """Load SBC rank histogram rows from a raw SBC output directory."""
    source_csv = Path(input_dir) / "rank_histogram.csv"
    if not source_csv.exists():
        raise FileNotFoundError(f"expected SBC rank histogram CSV at {source_csv}")
    with source_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"SBC rank histogram CSV has no rows: {source_csv}")

    required = {
        "method",
        "rank",
        "count",
        "num_replicates",
        "num_posterior_samples",
        "is_smoke_only",
        "evidence_note",
    }
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(f"SBC rank histogram CSV is missing columns: {', '.join(missing)}")
    rows = tuple(
        SBCRankHistogramRow(
            method=row["method"],
            rank=int(row["rank"]),
            count=int(row["count"]),
            num_replicates=int(row["num_replicates"]),
            num_posterior_samples=int(row["num_posterior_samples"]),
            is_smoke_only=_as_bool(row["is_smoke_only"]),
            evidence_note=row["evidence_note"],
        )
        for row in raw_rows
    )
    return SBCRankHistogramSummary(source_csv=source_csv, rows=rows)


def load_runtime_breakdown(input_dir: str | Path) -> RuntimeBreakdownSummary:
    """Load and aggregate optional runtime breakdown fields from ``summary.csv``."""
    source_csv = Path(input_dir) / "summary.csv"
    raw_rows = [
        row
        for row in _read_summary_rows(source_csv)
        if _as_bool(row.get("timing_breakdown_enabled", "false"))
    ]
    if not raw_rows:
        raise ValueError(f"summary CSV has no timing-breakdown rows: {source_csv}")

    required = {
        "method",
        "predictor",
        "bucket_size",
        "timing_cold_run_seconds",
        "timing_cold_compile_seconds",
        "timing_warm_iteration_seconds",
        "timing_planner_seconds",
        "timing_executor_seconds",
        "timing_gather_seconds",
        "timing_scatter_seconds",
        "timing_hvp_seconds",
        "timing_profiler_marker_count",
    }
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(f"runtime breakdown CSV is missing columns: {', '.join(missing)}")

    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = {}
    for row in raw_rows:
        key = (row["method"], row["predictor"], int(row["bucket_size"]))
        grouped.setdefault(key, []).append(row)

    rows = tuple(
        _aggregate_runtime_breakdown_group(method, predictor, bucket_size, group)
        for (method, predictor, bucket_size), group in sorted(grouped.items())
    )
    return RuntimeBreakdownSummary(source_csv=source_csv, rows=rows)


def has_trace_diagnostics(input_dir: str | Path) -> bool:
    """Return true when a raw output directory contains trace diagnostics."""
    return (Path(input_dir) / "diagnostics.csv").exists()


def has_sbc_rank_histogram(input_dir: str | Path) -> bool:
    """Return true when a raw output directory contains SBC rank histogram data."""
    return (Path(input_dir) / "rank_histogram.csv").exists()


def summary_has_timing_breakdown(input_dir: str | Path) -> bool:
    """Return true when a benchmark summary contains enabled timing-breakdown rows."""
    source_csv = Path(input_dir) / "summary.csv"
    with source_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        return False
    fieldnames = set(raw_rows[0])
    if not {"method", "predictor", "timing_breakdown_enabled"}.issubset(fieldnames):
        return False
    return any(_as_bool(row.get("timing_breakdown_enabled", "false")) for row in raw_rows)


def summary_has_oracle_gap(input_dir: str | Path) -> bool:
    """Return true when a benchmark summary contains oracle-gap scheduler rows."""
    source_csv = Path(input_dir) / "summary.csv"
    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames is not None and "scheduler_mode" in reader.fieldnames


def summary_has_chain_sweep(input_dir: str | Path) -> bool:
    """Return true when a benchmark summary contains more than one chain count."""
    source_csv = Path(input_dir) / "summary.csv"
    raw_rows = _read_summary_rows(source_csv)
    return len({int(row["num_chains"]) for row in raw_rows}) > 1


def summary_has_dimension_sweep(input_dir: str | Path) -> bool:
    """Return true when a benchmark summary contains more than one dimension."""
    source_csv = Path(input_dir) / "summary.csv"
    raw_rows = _read_summary_rows(source_csv)
    return len({int(row["dimension"]) for row in raw_rows}) > 1


def _read_summary_rows(source_csv: Path) -> list[dict[str, str]]:
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"benchmark summary CSV has no rows: {source_csv}")

    required = {
        "predictor",
        "bucket_size",
        "num_chains",
        "dimension",
        "t_mono",
        "t_bucket",
        "speedup",
        "bucket_padding_ratio",
        "bucket_divergence_count",
        "bucket_max_realized_tree_depth",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"benchmark summary CSV is missing columns: {', '.join(missing)}")
    return rows


def _read_oracle_gap_rows(source_csv: Path) -> list[dict[str, str]]:
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"oracle-gap summary CSV has no rows: {source_csv}")

    required = {
        "scheduler_mode",
        "scheduler_label",
        "bucket_size",
        "is_analysis_upper_bound",
        "t_mode",
        "speedup",
        "padding_ratio",
        "mean_predictor_abs_error",
        "mean_sum_bucket_realized_max",
        "mean_global_realized_max",
        "bucket_max_over_global_max",
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"oracle-gap summary CSV is missing columns: {', '.join(missing)}")
    return rows


def _read_predictor_calibration_rows(source_csv: Path) -> list[PredictorCalibrationRow]:
    with source_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"predictor calibration CSV has no rows: {source_csv}")

    required = {
        "scheduler_mode",
        "scheduler_label",
        "bucket_size",
        "is_analysis_upper_bound",
        "predicted_work",
        "realized_work",
        "abs_error",
    }
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(
            f"predictor calibration CSV is missing columns: {', '.join(missing)}"
        )
    return [
        PredictorCalibrationRow(
            scheduler_mode=row["scheduler_mode"],
            scheduler_label=row["scheduler_label"],
            bucket_size=int(row["bucket_size"]),
            is_analysis_upper_bound=_as_bool(row["is_analysis_upper_bound"]),
            predicted_work=float(row["predicted_work"]),
            realized_work=float(row["realized_work"]),
            abs_error=float(row["abs_error"]),
            is_summary_proxy=_as_bool(row.get("is_summary_proxy", "false")),
        )
        for row in raw_rows
    ]


def _read_padding_heatmap_rows(source_csv: Path) -> list[PaddingHeatmapRow]:
    with source_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"padding heatmap CSV has no rows: {source_csv}")

    required = {
        "scheduler_mode",
        "scheduler_label",
        "bucket_size",
        "is_analysis_upper_bound",
        "step",
        "bucket_index",
        "padding_count",
        "fill_fraction",
        "bucket_realized_max",
    }
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(f"padding heatmap CSV is missing columns: {', '.join(missing)}")
    return [
        PaddingHeatmapRow(
            scheduler_mode=row["scheduler_mode"],
            scheduler_label=row["scheduler_label"],
            bucket_size=int(row["bucket_size"]),
            is_analysis_upper_bound=_as_bool(row["is_analysis_upper_bound"]),
            step=int(row["step"]),
            bucket_index=int(row["bucket_index"]),
            padding_count=int(row["padding_count"]),
            fill_fraction=float(row["fill_fraction"]),
            bucket_realized_max=float(row["bucket_realized_max"]),
        )
        for row in raw_rows
    ]


def _read_speedup_heterogeneity_rows(source_csv: Path) -> list[SpeedupHeterogeneityRow]:
    with source_csv.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError(f"speedup heterogeneity CSV has no rows: {source_csv}")

    required = {
        "scheduler_mode",
        "scheduler_label",
        "bucket_size",
        "is_analysis_upper_bound",
        "speedup",
    }
    missing = sorted(required - set(raw_rows[0]))
    if missing:
        raise ValueError(
            f"speedup heterogeneity CSV is missing columns: {', '.join(missing)}"
        )
    return [
        SpeedupHeterogeneityRow(
            scheduler_mode=row["scheduler_mode"],
            scheduler_label=row["scheduler_label"],
            bucket_size=int(row["bucket_size"]),
            is_analysis_upper_bound=_as_bool(row["is_analysis_upper_bound"]),
            speedup=float(row["speedup"]),
            heterogeneity=_read_heterogeneity_value(row),
        )
        for row in raw_rows
    ]


def _fallback_predictor_calibration_rows(
    summary: OracleGapSummary,
) -> list[PredictorCalibrationRow]:
    rows: list[PredictorCalibrationRow] = []
    for row in summary.rows:
        if row.scheduler_mode == "monolithic":
            continue
        predicted_proxy = max(0.0, row.mean_global_realized_max - row.mean_predictor_abs_error)
        rows.append(
            PredictorCalibrationRow(
                scheduler_mode=row.scheduler_mode,
                scheduler_label=row.scheduler_label,
                bucket_size=row.bucket_size,
                is_analysis_upper_bound=row.is_analysis_upper_bound,
                predicted_work=predicted_proxy,
                realized_work=row.mean_global_realized_max,
                abs_error=row.mean_predictor_abs_error,
                is_summary_proxy=True,
            )
        )
    return rows


def _fallback_padding_heatmap_rows(summary: OracleGapSummary) -> list[PaddingHeatmapRow]:
    rows: list[PaddingHeatmapRow] = []
    for index, row in enumerate(summary.rows):
        if row.scheduler_mode == "monolithic":
            continue
        rows.append(
            PaddingHeatmapRow(
                scheduler_mode=row.scheduler_mode,
                scheduler_label=row.scheduler_label,
                bucket_size=row.bucket_size,
                is_analysis_upper_bound=row.is_analysis_upper_bound,
                step=0,
                bucket_index=index,
                padding_count=0,
                fill_fraction=max(0.0, 1.0 - row.mean_padding_ratio),
                bucket_realized_max=row.mean_global_realized_max,
            )
        )
    return rows


def _fallback_speedup_heterogeneity_rows(
    summary: OracleGapSummary,
) -> list[SpeedupHeterogeneityRow]:
    return [
        SpeedupHeterogeneityRow(
            scheduler_mode=row.scheduler_mode,
            scheduler_label=row.scheduler_label,
            bucket_size=row.bucket_size,
            is_analysis_upper_bound=row.is_analysis_upper_bound,
            speedup=row.mean_speedup,
            heterogeneity=max(0.0, row.mean_bucket_max_over_global_max - 1.0),
        )
        for row in summary.rows
    ]


def _aggregate_group(
    predictor: str,
    bucket_size: int,
    rows: list[dict[str, str]],
) -> AblationRow:
    return AblationRow(
        predictor=predictor,
        bucket_size=bucket_size,
        runs=len(rows),
        mean_t_mono=_mean(rows, "t_mono"),
        mean_t_bucket=_mean(rows, "t_bucket"),
        mean_speedup=_mean(rows, "speedup"),
        mean_padding_ratio=_mean(rows, "bucket_padding_ratio"),
        total_divergences=sum(int(row["bucket_divergence_count"]) for row in rows),
        max_tree_depth=max(int(row["bucket_max_realized_tree_depth"]) for row in rows),
    )


def _aggregate_chain_group(
    num_chains: int,
    predictor: str,
    bucket_size: int,
    rows: list[dict[str, str]],
) -> ChainScalingRow:
    return ChainScalingRow(
        num_chains=num_chains,
        predictor=predictor,
        bucket_size=bucket_size,
        runs=len(rows),
        mean_t_mono=_mean(rows, "t_mono"),
        mean_t_bucket=_mean(rows, "t_bucket"),
        mean_speedup=_mean(rows, "speedup"),
        mean_padding_ratio=_mean(rows, "bucket_padding_ratio"),
    )


def _aggregate_dimension_group(
    dimension: int,
    predictor: str,
    bucket_size: int,
    rows: list[dict[str, str]],
) -> DimensionScalingRow:
    return DimensionScalingRow(
        dimension=dimension,
        predictor=predictor,
        bucket_size=bucket_size,
        runs=len(rows),
        mean_t_mono=_mean(rows, "t_mono"),
        mean_t_bucket=_mean(rows, "t_bucket"),
        mean_speedup=_mean(rows, "speedup"),
        mean_padding_ratio=_mean(rows, "bucket_padding_ratio"),
    )


def _aggregate_oracle_gap_group(
    scheduler_mode: str,
    bucket_size: int,
    rows: list[dict[str, str]],
) -> OracleGapRow:
    planner_seconds = _optional_mean(rows, "timing_planner_seconds")
    executor_seconds = _optional_mean(rows, "timing_executor_seconds")
    gather_seconds = _optional_mean(rows, "timing_gather_seconds")
    scatter_seconds = _optional_mean(rows, "timing_scatter_seconds")
    gather_scatter_seconds = _optional_mean(
        rows,
        "timing_gather_scatter_seconds",
        default=gather_seconds + scatter_seconds,
    )
    hvp_seconds = _optional_mean(rows, "timing_hvp_seconds")
    warm_iteration_seconds = _optional_mean(
        rows,
        "timing_warm_iteration_seconds",
        default=_mean(rows, "t_mode"),
    )
    component_sum = (
        planner_seconds
        + executor_seconds
        + gather_scatter_seconds
        + hvp_seconds
    )
    unattributed_seconds = _optional_mean(
        rows,
        "timing_unattributed_seconds",
        default=max(0.0, warm_iteration_seconds - component_sum),
    )
    non_executor_overhead_seconds = _optional_mean(
        rows,
        "timing_non_executor_overhead_seconds",
        default=planner_seconds + gather_scatter_seconds + hvp_seconds + unattributed_seconds,
    )
    dominant_component, dominant_seconds = _dominant_timing_component(
        rows,
        planner_seconds=planner_seconds,
        executor_seconds=executor_seconds,
        gather_scatter_seconds=gather_scatter_seconds,
        hvp_seconds=hvp_seconds,
        unattributed_seconds=unattributed_seconds,
    )
    return OracleGapRow(
        scheduler_mode=scheduler_mode,
        scheduler_label=rows[0]["scheduler_label"],
        bucket_size=bucket_size,
        is_analysis_upper_bound=_as_bool(rows[0]["is_analysis_upper_bound"]),
        runs=len(rows),
        mean_t_mode=_mean(rows, "t_mode"),
        mean_speedup=_mean(rows, "speedup"),
        mean_padding_ratio=_mean(rows, "padding_ratio"),
        mean_predictor_abs_error=_mean(rows, "mean_predictor_abs_error"),
        mean_sum_bucket_realized_max=_mean(rows, "mean_sum_bucket_realized_max"),
        mean_global_realized_max=_mean(rows, "mean_global_realized_max"),
        mean_bucket_max_over_global_max=_mean(rows, "bucket_max_over_global_max"),
        mean_cold_compile_seconds=_optional_mean(rows, "timing_cold_compile_seconds"),
        mean_warm_iteration_seconds=warm_iteration_seconds,
        mean_planner_seconds=planner_seconds,
        mean_executor_seconds=executor_seconds,
        mean_gather_seconds=gather_seconds,
        mean_scatter_seconds=scatter_seconds,
        mean_gather_scatter_seconds=gather_scatter_seconds,
        mean_hvp_seconds=hvp_seconds,
        mean_unattributed_seconds=unattributed_seconds,
        mean_non_executor_overhead_seconds=non_executor_overhead_seconds,
        dominant_warm_component=dominant_component,
        dominant_warm_component_seconds=dominant_seconds,
    )


def _aggregate_runtime_breakdown_group(
    method: str,
    predictor: str,
    bucket_size: int,
    rows: list[dict[str, str]],
) -> RuntimeBreakdownRow:
    component_sum = (
        _mean(rows, "timing_planner_seconds")
        + _mean(rows, "timing_executor_seconds")
        + _mean(rows, "timing_gather_seconds")
        + _mean(rows, "timing_scatter_seconds")
        + _mean(rows, "timing_hvp_seconds")
    )
    mean_warm_iteration_seconds = _mean(rows, "timing_warm_iteration_seconds")
    return RuntimeBreakdownRow(
        method=method,
        predictor=predictor,
        bucket_size=bucket_size,
        runs=len(rows),
        mean_cold_run_seconds=_mean(rows, "timing_cold_run_seconds"),
        mean_cold_compile_seconds=_mean(rows, "timing_cold_compile_seconds"),
        mean_warm_iteration_seconds=mean_warm_iteration_seconds,
        mean_planner_seconds=_mean(rows, "timing_planner_seconds"),
        mean_executor_seconds=_mean(rows, "timing_executor_seconds"),
        mean_gather_seconds=_mean(rows, "timing_gather_seconds"),
        mean_scatter_seconds=_mean(rows, "timing_scatter_seconds"),
        mean_hvp_seconds=_mean(rows, "timing_hvp_seconds"),
        mean_unattributed_seconds=max(0.0, mean_warm_iteration_seconds - component_sum),
        mean_profiler_marker_count=_mean(rows, "timing_profiler_marker_count"),
    )


def _mean(rows: list[dict[str, str]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _optional_mean(
    rows: list[dict[str, str]],
    key: str,
    *,
    default: float = 0.0,
) -> float:
    values = [
        float(row[key])
        for row in rows
        if key in row and row[key] not in {"", "None", "none"}
    ]
    if not values:
        return default
    return sum(values) / len(values)


def _dominant_timing_component(
    rows: list[dict[str, str]],
    *,
    planner_seconds: float,
    executor_seconds: float,
    gather_scatter_seconds: float,
    hvp_seconds: float,
    unattributed_seconds: float,
) -> tuple[str, float]:
    if not any(_as_bool(row.get("timing_breakdown_enabled", "false")) for row in rows):
        return "unavailable", 0.0
    components = {
        "planner": planner_seconds,
        "executor": executor_seconds,
        "gather_scatter": gather_scatter_seconds,
        "hvp": hvp_seconds,
        "unattributed": unattributed_seconds,
    }
    return max(components.items(), key=lambda item: item[1])


def _sort_key(item: tuple[tuple[str, int], list[dict[str, str]]]) -> tuple[str, int]:
    (predictor, bucket_size), _rows = item
    return predictor, bucket_size


def _oracle_gap_sort_key(
    item: tuple[tuple[str, int], list[dict[str, str]]],
) -> tuple[int, int]:
    (scheduler_mode, bucket_size), _rows = item
    order = {
        "monolithic": 0,
        "unsorted": 1,
        "random": 2,
        "history": 3,
        "hvp": 4,
        "hybrid": 5,
        "oracle_previous": 6,
        "oracle_current": 7,
    }
    return order.get(scheduler_mode, 99), bucket_size


def _as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def _read_heterogeneity_value(row: dict[str, str]) -> float:
    if "realized_depth_cv" in row:
        return float(row["realized_depth_cv"])
    if "heterogeneity" in row:
        return float(row["heterogeneity"])
    if "bucket_max_over_global_max" in row:
        return max(0.0, float(row["bucket_max_over_global_max"]) - 1.0)
    return 0.0
