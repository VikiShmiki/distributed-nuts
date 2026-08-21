"""Trace-level posterior diagnostics for correctness QA outputs."""

from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TraceDiagnosticRow:
    """One diagnostic row for one method and one position dimension."""

    method: str
    variable: str
    dimension: int
    mean: float
    sd: float
    r_hat: float
    bulk_ess: float
    tail_ess: float
    mcse_mean: float
    divergence_count: int
    max_tree_depth_hit_count: int
    max_realized_tree_depth: int


class DiagnosticsDependencyError(RuntimeError):
    """Raised when the optional diagnostics dependency is unavailable."""


def summarize_trace_diagnostics(
    *,
    method: str,
    trace_positions: Any,
    transition_info: Any,
) -> tuple[TraceDiagnosticRow, ...]:
    """Compute ArviZ diagnostics for a saved position trace.

    ``trace_positions`` must have shape ``(draw, chain, dimension)``. ArviZ
    expects ``(chain, draw)`` arrays for each scalar variable, so this function
    transposes each position dimension into its own variable.
    """
    az = _import_arviz()
    positions = np.asarray(trace_positions)
    if positions.ndim != 3:
        raise ValueError(
            "trace_positions must have shape (draw, chain, dimension); "
            f"got {positions.shape}"
        )
    if positions.shape[0] <= 1:
        raise ValueError("trace diagnostics require at least two saved draws")

    posterior = {
        _variable_name(dim): np.swapaxes(positions[:, :, dim], 0, 1)
        for dim in range(positions.shape[2])
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        inference_data = az.from_dict(posterior=posterior)
        r_hat = az.rhat(inference_data, method="rank")
        bulk_ess = az.ess(inference_data, method="bulk")
        tail_ess = az.ess(inference_data, method="tail")
        mcse_mean = az.mcse(inference_data, method="mean")

    divergence_count = int(np.sum(np.asarray(transition_info.divergence_flag)))
    max_tree_depth_hit_count = int(np.sum(np.asarray(transition_info.max_tree_depth_hit)))
    max_realized_tree_depth = int(np.max(np.asarray(transition_info.realized_tree_depth)))

    rows: list[TraceDiagnosticRow] = []
    for dim in range(positions.shape[2]):
        variable = _variable_name(dim)
        values = positions[:, :, dim]
        rows.append(
            TraceDiagnosticRow(
                method=method,
                variable=variable,
                dimension=dim,
                mean=float(np.mean(values)),
                sd=float(np.std(values, ddof=1)),
                r_hat=_dataset_scalar(r_hat, variable),
                bulk_ess=_dataset_scalar(bulk_ess, variable),
                tail_ess=_dataset_scalar(tail_ess, variable),
                mcse_mean=_dataset_scalar(mcse_mean, variable),
                divergence_count=divergence_count,
                max_tree_depth_hit_count=max_tree_depth_hit_count,
                max_realized_tree_depth=max_realized_tree_depth,
            )
        )
    return tuple(rows)


def diagnostics_rows_to_csv(rows: tuple[TraceDiagnosticRow, ...]) -> str:
    """Serialize trace diagnostics as CSV."""
    if not rows:
        return ""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].__dict__), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row.__dict__)
    return buffer.getvalue()


def diagnostics_rows_equal(
    left: tuple[TraceDiagnosticRow, ...],
    right: tuple[TraceDiagnosticRow, ...],
) -> bool:
    """Return true when two method-level diagnostic row groups match exactly."""
    if len(left) != len(right):
        return False
    for left_row, right_row in zip(left, right, strict=True):
        if left_row.dimension != right_row.dimension or left_row.variable != right_row.variable:
            return False
        numeric_fields = (
            "mean",
            "sd",
            "r_hat",
            "bulk_ess",
            "tail_ess",
            "mcse_mean",
        )
        for field in numeric_fields:
            left_value = getattr(left_row, field)
            right_value = getattr(right_row, field)
            if math.isnan(left_value) and math.isnan(right_value):
                continue
            if left_value != right_value:
                return False
        count_fields = (
            "divergence_count",
            "max_tree_depth_hit_count",
            "max_realized_tree_depth",
        )
        for field in count_fields:
            if getattr(left_row, field) != getattr(right_row, field):
                return False
    return True


def load_diagnostics_csv(input_dir: str | Path) -> tuple[TraceDiagnosticRow, ...]:
    """Load a raw trace diagnostics CSV from ``input_dir``."""
    source_csv = Path(input_dir) / "diagnostics.csv"
    with source_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"diagnostics CSV has no rows: {source_csv}")

    required = set(TraceDiagnosticRow.__dataclass_fields__)
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"diagnostics CSV is missing columns: {', '.join(missing)}")

    return tuple(
        TraceDiagnosticRow(
            method=row["method"],
            variable=row["variable"],
            dimension=int(row["dimension"]),
            mean=float(row["mean"]),
            sd=float(row["sd"]),
            r_hat=float(row["r_hat"]),
            bulk_ess=float(row["bulk_ess"]),
            tail_ess=float(row["tail_ess"]),
            mcse_mean=float(row["mcse_mean"]),
            divergence_count=int(row["divergence_count"]),
            max_tree_depth_hit_count=int(row["max_tree_depth_hit_count"]),
            max_realized_tree_depth=int(row["max_realized_tree_depth"]),
        )
        for row in rows
    )


def _import_arviz() -> Any:
    try:
        import arviz as az
    except ModuleNotFoundError as exc:
        raise DiagnosticsDependencyError(
            "Trace diagnostics require ArviZ. Install project dependencies with "
            "`python -m pip install -e '.[dev]'` or install `arviz` directly."
        ) from exc
    return az


def _variable_name(dimension: int) -> str:
    return f"q_{dimension}"


def _dataset_scalar(dataset: Any, variable: str) -> float:
    return float(np.asarray(dataset[variable]).reshape(()))
