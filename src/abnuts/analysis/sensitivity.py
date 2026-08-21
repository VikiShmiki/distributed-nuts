"""Sensitivity and worst-group aggregation for existing benchmark outputs."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from abnuts.analysis.diagnostics import TraceDiagnosticRow, load_diagnostics_csv


@dataclass(frozen=True)
class SensitivityObservation:
    """One speedup row normalized from a raw benchmark summary."""

    source_csv: Path
    model: str
    model_family: str
    parameterization: str
    predictor_mode: str
    bucket_size: int
    num_chains: int
    dimension: int
    num_steps: int
    seed: int
    speedup: float
    padding_ratio: float
    divergence_count: int
    max_tree_depth: int
    is_analysis_upper_bound: bool
    used_model_family_fallback: bool


@dataclass(frozen=True)
class SensitivityStratumRow:
    """Worst observed group for one requested sensitivity stratum."""

    stratum: str
    group: str
    runs: int
    source_count: int
    mean_speedup: float
    worst_speedup: float
    mean_padding_ratio: float
    total_divergences: int
    max_tree_depth: int
    analysis_upper_bound_runs: int


@dataclass(frozen=True)
class SensitivityCoverageRow:
    """Coverage note for a missing or collapsed requested stratum."""

    stratum: str
    status: str
    note: str


@dataclass(frozen=True)
class SensitivitySummary:
    """Sensitivity table data and evidence-quality metadata."""

    source_root: Path
    source_csvs: tuple[Path, ...]
    observation_count: int
    diagnostic_file_count: int
    diagnostic_row_count: int
    total_divergences: int
    max_tree_depth: int
    rows: tuple[SensitivityStratumRow, ...]
    coverage_rows: tuple[SensitivityCoverageRow, ...]
    is_smoke_only: bool
    evidence_note: str
    max_r_hat: float | None
    min_bulk_ess: float | None
    min_tail_ess: float | None

    @property
    def worst_row(self) -> SensitivityStratumRow:
        """Return the lowest mean-speedup group across requested strata."""
        if not self.rows:
            raise ValueError("cannot choose a worst group from an empty summary")
        return min(self.rows, key=lambda row: (row.mean_speedup, row.worst_speedup))


def load_sensitivity(input_dir: str | Path) -> SensitivitySummary:
    """Load sensitivity and worst-group data from a raw result tree."""
    source_root = Path(input_dir)
    observations = _load_speedup_observations(source_root)
    if not observations:
        raise ValueError(f"no bucketed speedup rows found under {source_root}")

    diagnostics = _load_diagnostics(source_root)
    depth_labels = _depth_quantile_labels(observations)
    rows = _worst_group_rows(observations, depth_labels)
    coverage_rows = _coverage_rows(observations, diagnostics, depth_labels)
    source_csvs = tuple(sorted({row.source_csv for row in observations}))
    is_smoke_only, evidence_note = _evidence_note(observations, source_root)

    return SensitivitySummary(
        source_root=source_root,
        source_csvs=source_csvs,
        observation_count=len(observations),
        diagnostic_file_count=len({row.source_csv for row in diagnostics}),
        diagnostic_row_count=len(diagnostics),
        total_divergences=sum(row.divergence_count for row in observations),
        max_tree_depth=max(row.max_tree_depth for row in observations),
        rows=rows,
        coverage_rows=coverage_rows,
        is_smoke_only=is_smoke_only,
        evidence_note=evidence_note,
        max_r_hat=_optional_max(row.r_hat for row in diagnostics),
        min_bulk_ess=_optional_min(row.bulk_ess for row in diagnostics),
        min_tail_ess=_optional_min(row.tail_ess for row in diagnostics),
    )


def _load_speedup_observations(source_root: Path) -> list[SensitivityObservation]:
    observations: list[SensitivityObservation] = []
    for source_csv in sorted(source_root.rglob("summary.csv")):
        with source_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue

        manifest = _read_manifest(source_csv.parent)
        if "scheduler_mode" in rows[0]:
            observations.extend(_oracle_gap_observations(source_csv, rows, manifest))
        else:
            observations.extend(_standard_observations(source_csv, rows, manifest))
    return observations


def _standard_observations(
    source_csv: Path,
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> list[SensitivityObservation]:
    observations: list[SensitivityObservation] = []
    for row in rows:
        method = row.get("method", "bucketed") or "bucketed"
        if method != "bucketed":
            continue
        speedup_text = _first_non_empty(row, ("speedup", "speedup_vs_monolithic"))
        if speedup_text is None:
            continue
        model_family, used_model_family_fallback = _model_family(row, manifest)
        observations.append(
            SensitivityObservation(
                source_csv=source_csv,
                model=_first_non_empty(row, ("model",)) or "unknown",
                model_family=model_family,
                parameterization=_metadata_value(row, manifest, "parameterization"),
                predictor_mode=_first_non_empty(row, ("predictor",)) or "unknown",
                bucket_size=_optional_int(row, "bucket_size", default=0),
                num_chains=_optional_int(row, "num_chains", default=0),
                dimension=_optional_int(row, "dimension", default=0),
                num_steps=_optional_int(row, "num_steps", default=0),
                seed=_optional_int(row, "seed", default=0),
                speedup=float(speedup_text),
                padding_ratio=_optional_float(row, "bucket_padding_ratio", default=0.0),
                divergence_count=_optional_int(
                    row,
                    "bucket_divergence_count",
                    fallback_key="method_divergence_count",
                    default=0,
                ),
                max_tree_depth=_optional_int(
                    row,
                    "bucket_max_realized_tree_depth",
                    fallback_key="method_max_realized_tree_depth",
                    default=0,
                ),
                is_analysis_upper_bound=False,
                used_model_family_fallback=used_model_family_fallback,
            )
        )
    return observations


def _oracle_gap_observations(
    source_csv: Path,
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> list[SensitivityObservation]:
    observations: list[SensitivityObservation] = []
    for row in rows:
        scheduler_mode = row.get("scheduler_mode", "")
        if scheduler_mode == "monolithic":
            continue
        model_family, used_model_family_fallback = _model_family(row, manifest)
        is_upper_bound = _as_bool(row.get("is_analysis_upper_bound", "false"))
        scheduler_label = row.get("scheduler_label", scheduler_mode) or scheduler_mode
        if is_upper_bound:
            scheduler_label = f"{scheduler_label} (analysis-only upper bound)"
        observations.append(
            SensitivityObservation(
                source_csv=source_csv,
                model=_first_non_empty(row, ("model",)) or "unknown",
                model_family=model_family,
                parameterization=_metadata_value(row, manifest, "parameterization"),
                predictor_mode=scheduler_label,
                bucket_size=_optional_int(row, "bucket_size", default=0),
                num_chains=_optional_int(row, "num_chains", default=0),
                dimension=_optional_int(row, "dimension", default=0),
                num_steps=_optional_int(row, "num_steps", default=0),
                seed=_optional_int(row, "seed", default=0),
                speedup=_optional_float(row, "speedup", default=0.0),
                padding_ratio=_optional_float(row, "padding_ratio", default=0.0),
                divergence_count=_optional_int(row, "divergence_count", default=0),
                max_tree_depth=_optional_int(row, "max_realized_tree_depth", default=0),
                is_analysis_upper_bound=is_upper_bound,
                used_model_family_fallback=used_model_family_fallback,
            )
        )
    return observations


def _load_diagnostics(source_root: Path) -> list[TraceDiagnosticRowWithSource]:
    diagnostics: list[TraceDiagnosticRowWithSource] = []
    for diagnostics_csv in sorted(source_root.rglob("diagnostics.csv")):
        try:
            rows = load_diagnostics_csv(diagnostics_csv.parent)
        except (OSError, ValueError):
            continue
        diagnostics.extend(
            TraceDiagnosticRowWithSource(source_csv=diagnostics_csv, row=row) for row in rows
        )
    return diagnostics


@dataclass(frozen=True)
class TraceDiagnosticRowWithSource:
    """Trace diagnostic row with its source path attached."""

    source_csv: Path
    row: TraceDiagnosticRow

    @property
    def r_hat(self) -> float:
        """Return the diagnostic R-hat."""
        return self.row.r_hat

    @property
    def bulk_ess(self) -> float:
        """Return the bulk ESS."""
        return self.row.bulk_ess

    @property
    def tail_ess(self) -> float:
        """Return the tail ESS."""
        return self.row.tail_ess


def _worst_group_rows(
    observations: list[SensitivityObservation],
    depth_labels: dict[int, str],
) -> tuple[SensitivityStratumRow, ...]:
    requested: tuple[tuple[str, Callable[[SensitivityObservation], str]], ...] = (
        ("model family", lambda row: row.model_family),
        ("dimension", lambda row: f"D={row.dimension}" if row.dimension else "missing"),
        ("chain count", lambda row: f"C={row.num_chains}" if row.num_chains else "missing"),
        ("depth quantile", lambda row: depth_labels[id(row)]),
        (
            "divergence status",
            lambda row: "divergent" if row.divergence_count > 0 else "no divergences",
        ),
        ("predictor mode", lambda row: row.predictor_mode),
    )

    worst_rows: list[SensitivityStratumRow] = []
    for stratum, group_fn in requested:
        grouped: dict[str, list[SensitivityObservation]] = {}
        for observation in observations:
            grouped.setdefault(group_fn(observation), []).append(observation)
        candidate_rows = [
            _summarize_group(stratum, group, group_observations)
            for group, group_observations in grouped.items()
        ]
        worst_rows.append(
            min(candidate_rows, key=lambda row: (row.mean_speedup, row.worst_speedup, row.group))
        )
    return tuple(worst_rows)


def _summarize_group(
    stratum: str,
    group: str,
    observations: list[SensitivityObservation],
) -> SensitivityStratumRow:
    return SensitivityStratumRow(
        stratum=stratum,
        group=group,
        runs=len(observations),
        source_count=len({row.source_csv for row in observations}),
        mean_speedup=sum(row.speedup for row in observations) / len(observations),
        worst_speedup=min(row.speedup for row in observations),
        mean_padding_ratio=sum(row.padding_ratio for row in observations) / len(observations),
        total_divergences=sum(row.divergence_count for row in observations),
        max_tree_depth=max(row.max_tree_depth for row in observations),
        analysis_upper_bound_runs=sum(row.is_analysis_upper_bound for row in observations),
    )


def _coverage_rows(
    observations: list[SensitivityObservation],
    diagnostics: list[TraceDiagnosticRowWithSource],
    depth_labels: dict[int, str],
) -> tuple[SensitivityCoverageRow, ...]:
    coverage: list[SensitivityCoverageRow] = []
    _append_group_coverage(
        coverage,
        "model family",
        [row.model_family for row in observations],
    )
    _append_group_coverage(
        coverage,
        "dimension",
        [f"D={row.dimension}" for row in observations if row.dimension],
    )
    _append_group_coverage(
        coverage,
        "chain count",
        [f"C={row.num_chains}" for row in observations if row.num_chains],
    )
    _append_group_coverage(
        coverage,
        "depth quantile",
        [depth_labels[id(row)] for row in observations],
        collapsed_note="Depth quantiles are collapsed because all included rows have the same "
        "maximum realized tree depth.",
    )
    divergence_statuses = {
        "divergent" if row.divergence_count > 0 else "no divergences"
        for row in observations
    }
    if "divergent" not in divergence_statuses:
        coverage.append(
            SensitivityCoverageRow(
                stratum="divergence status",
                status="missing divergent group",
                note="No included tiny summary row reported divergences.",
            )
        )
    _append_group_coverage(
        coverage,
        "predictor mode",
        [row.predictor_mode for row in observations],
    )

    fallback_count = sum(row.used_model_family_fallback for row in observations)
    if fallback_count:
        coverage.append(
            SensitivityCoverageRow(
                stratum="model family metadata",
                status="partial",
                note=f"{fallback_count} rows used the model name as the model-family fallback.",
            )
        )

    if diagnostics:
        coverage.append(
            SensitivityCoverageRow(
                stratum="posterior diagnostics",
                status="partial",
                note=(
                    f"Loaded {len(diagnostics)} diagnostic rows from "
                    f"{len({row.source_csv for row in diagnostics})} diagnostics.csv files; "
                    "speedup summaries still lack ESS/s fields."
                ),
            )
        )
    else:
        coverage.append(
            SensitivityCoverageRow(
                stratum="posterior diagnostics",
                status="missing",
                note="No diagnostics.csv files were found under the input root.",
            )
        )
    return tuple(coverage)


def _append_group_coverage(
    coverage: list[SensitivityCoverageRow],
    stratum: str,
    labels: list[str],
    *,
    collapsed_note: str | None = None,
) -> None:
    unique = sorted({label for label in labels if label})
    if not unique:
        coverage.append(
            SensitivityCoverageRow(
                stratum=stratum,
                status="missing",
                note=f"No included speedup rows had a usable {stratum} value.",
            )
        )
    elif len(unique) == 1:
        coverage.append(
            SensitivityCoverageRow(
                stratum=stratum,
                status="collapsed",
                note=collapsed_note or f"Only one observed group: {unique[0]}.",
            )
        )


def _depth_quantile_labels(
    observations: list[SensitivityObservation],
) -> dict[int, str]:
    depths = sorted(row.max_tree_depth for row in observations)
    if len(set(depths)) <= 1:
        depth = depths[0] if depths else 0
        return {id(row): f"all depth {depth}" for row in observations}

    labels: dict[int, str] = {}
    total = len(depths)
    for row in observations:
        less_count = sum(depth < row.max_tree_depth for depth in depths)
        quantile = min(4, int((less_count / total) * 4) + 1)
        labels[id(row)] = f"Q{quantile} (depth {row.max_tree_depth})"
    return labels


def _evidence_note(
    observations: list[SensitivityObservation],
    source_root: Path,
) -> tuple[bool, str]:
    seed_count = len({row.seed for row in observations})
    max_steps = max(row.num_steps for row in observations)
    max_chains = max(row.num_chains for row in observations)
    has_tiny_path = any("tiny" in part for part in source_root.parts)
    is_smoke_only = has_tiny_path or seed_count < 4 or max_steps < 1000 or max_chains < 32
    if is_smoke_only:
        return (
            True,
            "Smoke-only evidence: existing rows are too small for paper-scale claims "
            f"(seeds={seed_count}, max steps={max_steps}, max chains={max_chains}).",
        )
    return (
        False,
        "Paper-scale candidate evidence: verify paired uncertainty before making claims "
        f"(seeds={seed_count}, max steps={max_steps}, max chains={max_chains}).",
    )


def _read_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _model_family(row: dict[str, str], manifest: dict[str, Any]) -> tuple[str, bool]:
    row_value = row.get("model_family", "").strip()
    if row_value:
        return row_value, False
    manifest_value = _manifest_model_metadata(manifest).get("model_family", "")
    if isinstance(manifest_value, str) and manifest_value:
        return manifest_value, False
    model = row.get("model", "").strip() or "unknown"
    return model, True


def _metadata_value(row: dict[str, str], manifest: dict[str, Any], key: str) -> str:
    row_value = row.get(key, "").strip()
    if row_value:
        return row_value
    manifest_value = _manifest_model_metadata(manifest).get(key, "")
    return manifest_value if isinstance(manifest_value, str) else ""


def _manifest_model_metadata(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("model_metadata", {})
    return value if isinstance(value, dict) else {}


def _first_non_empty(row: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and value.strip() != "":
            return value
    return None


def _optional_int(
    row: dict[str, str],
    key: str,
    *,
    fallback_key: str | None = None,
    default: int,
) -> int:
    value = _first_non_empty(row, (key,))
    if value is None and fallback_key is not None:
        value = _first_non_empty(row, (fallback_key,))
    if value is None:
        return default
    return int(float(value))


def _optional_float(row: dict[str, str], key: str, *, default: float) -> float:
    value = _first_non_empty(row, (key,))
    if value is None:
        return default
    return float(value)


def _as_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes"}


def _optional_max(values: Any) -> float | None:
    values = tuple(float(value) for value in values)
    if not values:
        return None
    return max(values)


def _optional_min(values: Any) -> float | None:
    values = tuple(float(value) for value in values)
    if not values:
        return None
    return min(values)
