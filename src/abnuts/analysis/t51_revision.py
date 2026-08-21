"""Generate the T51 statistical-revision artifacts from retained CPU timings."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BOOTSTRAP_DRAWS = 10_000


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-input", type=Path, required=True)
    parser.add_argument("--scaling-input", type=Path, required=True)
    parser.add_argument("--predictor-input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _read_summaries(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("summary.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["_source"] = str(path)
                rows.append(row)
    if not rows:
        raise ValueError(f"no summary.csv files found under {root}")
    return rows


def _json_times(row: dict[str, str], field: str) -> np.ndarray:
    values = np.asarray(json.loads(row[field]), dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid timing sample in {row['_source']}: {field}")
    return values


def _display_predictor(value: str) -> str:
    return {"none": "unsorted", "last_depth": "last-depth"}.get(value, value)


def _aggregate_timing_rows(
    rows: list[dict[str, str]],
    *,
    include_chains: bool,
) -> list[dict[str, Any]]:
    mono: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        if row["method"] == "monolithic":
            mono[(row["model"], int(row["num_chains"]), int(row["seed"]))] = row

    grouped: dict[tuple[Any, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] != "bucketed":
            continue
        key: tuple[Any, ...] = (
            row["model"],
            int(row["num_chains"]) if include_chains else 0,
            row["predictor"],
            int(row["bucket_size"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        model, chains, predictor, bucket_size = key
        group = sorted(group, key=lambda row: int(row["seed"]))
        if len(group) != 5:
            raise ValueError(f"expected five seeds for {key}, found {len(group)}")
        seed_speedups: list[float] = []
        all_method_times: list[float] = []
        for row in group:
            mono_row = mono[(model, int(row["num_chains"]), int(row["seed"]))]
            seed_speedups.append(
                float(mono_row["timing_warm_median_seconds"])
                / float(row["timing_warm_median_seconds"])
            )
            all_method_times.extend(_json_times(row, "timing_warm_repeat_seconds_json"))
        ci_low, ci_high = _hierarchical_speedup_ci(group, mono)
        discrete_match = all(
            int(row[mono_field]) == int(row[bucket_field])
            for row in group
            for mono_field, bucket_field in (
                ("mono_total_leapfrog_count", "bucket_total_leapfrog_count"),
                ("mono_divergence_count", "bucket_divergence_count"),
                ("mono_max_realized_tree_depth", "bucket_max_realized_tree_depth"),
            )
        )
        output.append(
            {
                "model": model,
                "num_chains": int(group[0]["num_chains"]),
                "predictor": _display_predictor(predictor),
                "bucket_size": bucket_size,
                "seeds": len(group),
                "repeats_per_seed": len(
                    _json_times(group[0], "timing_warm_repeat_seconds_json")
                ),
                "median_speedup": float(np.median(seed_speedups)),
                "speedup_seed_min": min(seed_speedups),
                "speedup_seed_max": max(seed_speedups),
                "speedup_ci_low": ci_low,
                "speedup_ci_high": ci_high,
                "median_time_seconds": float(np.median(all_method_times)),
                "time_q1_seconds": float(np.percentile(all_method_times, 25)),
                "time_q3_seconds": float(np.percentile(all_method_times, 75)),
                "all_transition_metrics_match": discrete_match,
            }
        )
    return output


def _hierarchical_speedup_ci(
    group: list[dict[str, str]],
    mono: dict[tuple[str, int, int], dict[str, str]],
) -> tuple[float, float]:
    rng = np.random.default_rng(20260810)
    seed_bootstraps: list[np.ndarray] = []
    for row in group:
        mono_row = mono[(row["model"], int(row["num_chains"]), int(row["seed"]))]
        method_times = _json_times(row, "timing_warm_repeat_seconds_json")
        mono_times = _json_times(mono_row, "timing_warm_repeat_seconds_json")
        method_draws = rng.choice(
            method_times,
            size=(BOOTSTRAP_DRAWS, method_times.size),
            replace=True,
        )
        mono_draws = rng.choice(
            mono_times,
            size=(BOOTSTRAP_DRAWS, mono_times.size),
            replace=True,
        )
        seed_bootstraps.append(
            np.median(mono_draws, axis=1) / np.median(method_draws, axis=1)
        )
    speedup_by_draw_seed = np.stack(seed_bootstraps, axis=1)
    sampled_seed_indices = rng.integers(
        0,
        len(group),
        size=(BOOTSTRAP_DRAWS, len(group)),
    )
    sampled_speedups = np.take_along_axis(
        speedup_by_draw_seed,
        sampled_seed_indices,
        axis=1,
    )
    estimates = np.median(sampled_speedups, axis=1)
    return tuple(float(value) for value in np.percentile(estimates, [2.5, 97.5]))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _predictor_diagnostics(root: Path) -> list[dict[str, Any]]:
    path = root / "predictor_calibration.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["seed"]), row["scheduler_mode"], int(row["bucket_size"]))].append(row)

    per_seed: list[dict[str, Any]] = []
    for (seed, mode, width), group in sorted(grouped.items()):
        by_step: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in group:
            by_step[int(row["step"])].append(row)
        correlations: list[float] = []
        recalls: list[float] = []
        spreads: list[float] = []
        absolute_errors: list[float] = []
        actual_matrix: list[np.ndarray] = []
        for step_rows in by_step.values():
            step_rows.sort(key=lambda row: int(row["chain"]))
            predicted = np.asarray([float(row["predicted_work"]) for row in step_rows])
            actual = np.asarray([float(row["realized_work"]) for row in step_rows])
            actual_matrix.append(actual)
            correlations.append(_spearman(predicted, actual))
            k = max(1, int(np.ceil(0.10 * actual.size)))
            actual_tail = set(np.argsort(actual, kind="mergesort")[-k:])
            predicted_tail = set(np.argsort(predicted, kind="mergesort")[-k:])
            recalls.append(len(actual_tail & predicted_tail) / k)
            sorted_actual = actual[np.argsort(predicted, kind="mergesort")]
            spreads.extend(
                float(np.max(chunk) - np.min(chunk))
                for chunk in np.array_split(sorted_actual, actual.size // width)
            )
            absolute_errors.extend(np.abs(predicted - actual))
        actual_by_step = np.stack(actual_matrix)
        lag = float(np.corrcoef(actual_by_step[:-1].ravel(), actual_by_step[1:].ravel())[0, 1])
        per_seed.append(
            {
                "seed": seed,
                "predictor": "last-depth" if mode == "oracle_previous" else mode.replace("_", "-"),
                "bucket_size": width,
                "mean_step_spearman": float(np.mean(correlations)),
                "deepest_10pct_recall": float(np.mean(recalls)),
                "mean_within_bucket_depth_range": float(np.mean(spreads)),
                "mean_absolute_error": float(np.mean(absolute_errors)),
                "depth_lag1_correlation": lag,
            }
        )

    aggregate: list[dict[str, Any]] = []
    grouped_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped_seed[(row["predictor"], row["bucket_size"])].append(row)
    for (predictor, width), group in sorted(grouped_seed.items()):
        aggregate.append(
            {
                "predictor": predictor,
                "bucket_size": width,
                "seeds": len(group),
                **{
                    field: float(np.mean([row[field] for row in group]))
                    for field in (
                        "mean_step_spearman",
                        "deepest_10pct_recall",
                        "mean_within_bucket_depth_range",
                        "mean_absolute_error",
                        "depth_lag1_correlation",
                    )
                },
            }
        )
    return aggregate


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing to write empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_latex(
    out: Path,
    final_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    predictor_rows: list[dict[str, Any]],
) -> None:
    best: dict[str, dict[str, Any]] = {}
    for row in final_rows:
        if row["model"] not in best or row["median_speedup"] > best[row["model"]]["median_speedup"]:
            best[row["model"]] = row
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Median speedup & 95\% CI & Predictor & Width \\",
        r"\midrule",
    ]
    for model, row in sorted(best.items()):
        label = model.replace("_", r"\_")
        lines.append(
            f"{label} & {row['median_speedup']:.3f} & "
            f"[{row['speedup_ci_low']:.3f}, {row['speedup_ci_high']:.3f}] & "
            f"{row['predictor']} & {row['bucket_size']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "table_t51_best_models.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Predictor & Width & Spearman & Tail recall & Bucket range & MAE \\",
        r"\midrule",
    ]
    for row in predictor_rows:
        lines.append(
            f"{row['predictor']} & {row['bucket_size']} & "
            f"{row['mean_step_spearman']:.3f} & {row['deepest_10pct_recall']:.3f} & "
            f"{row['mean_within_bucket_depth_range']:.3f} & {row['mean_absolute_error']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "table_t51_predictor_diagnostics.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Model & Chains & Median speedup & 95\% CI & Policy/width \\",
        r"\midrule",
    ]
    for model in sorted({row["model"] for row in scaling_rows}):
        for chains in (128, 256, 512, 1024):
            candidates = [
                row for row in scaling_rows
                if row["model"] == model and row["num_chains"] == chains
            ]
            row = max(candidates, key=lambda item: item["median_speedup"])
            label = model.replace("_", r"\_")
            lines.append(
                f"{label} & {chains} & {row['median_speedup']:.3f} & "
                f"[{row['speedup_ci_low']:.3f}, {row['speedup_ci_high']:.3f}] & "
                f"{row['predictor']}/{row['bucket_size']} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "table_t51_scaling.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(final_input: Path, scaling_input: Path, predictor_input: Path, out: Path) -> None:
    final_rows = _aggregate_timing_rows(_read_summaries(final_input), include_chains=False)
    scaling_rows = _aggregate_timing_rows(_read_summaries(scaling_input), include_chains=True)
    predictor_rows = _predictor_diagnostics(predictor_input)
    bucket_speedups = np.asarray([row["median_speedup"] for row in final_rows])
    headline = {
        "model_count": len({row["model"] for row in final_rows}),
        "bucketed_configurations": len(final_rows),
        "configurations_faster_than_monolithic": int(np.sum(bucket_speedups > 1.0)),
        "fraction_faster": float(np.mean(bucket_speedups > 1.0)),
        "median_speedup": float(np.median(bucket_speedups)),
        "best_speedup": float(np.max(bucket_speedups)),
        "worst_speedup": float(np.min(bucket_speedups)),
        "all_transition_metrics_match": all(
            row["all_transition_metrics_match"] for row in final_rows
        ),
        "seeds": 5,
        "warm_repeats_per_seed": 30,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "final_config_summary.csv", final_rows)
    _write_csv(out / "scaling_summary.csv", scaling_rows)
    _write_csv(out / "predictor_diagnostics.csv", predictor_rows)
    (out / "headline.json").write_text(json.dumps(headline, indent=2) + "\n", encoding="utf-8")
    report = ["# T51 preferred statistical protocol", "", f"- Seeds: 5", "- Blocked warm repetitions per seed/configuration: 30", f"- Bucketed configurations: {headline['bucketed_configurations']}", f"- Faster than monolithic: {headline['configurations_faster_than_monolithic']}/{headline['bucketed_configurations']}", f"- Median/best/worst speedup: {headline['median_speedup']:.3f}x / {headline['best_speedup']:.3f}x / {headline['worst_speedup']:.3f}x", f"- Leapfrog/divergence/max-depth parity: {headline['all_transition_metrics_match']}", "", "Intervals use a deterministic hierarchical bootstrap: sampler seeds and the retained monolithic/method timing samples are independently resampled; the reported statistic is the median speedup across seeds. Repeat indices are not treated as paired because methods were timed sequentially."]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _write_latex(out, final_rows, scaling_rows, predictor_rows)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output is not empty: {args.out}; pass --overwrite")
    generate(args.final_input, args.scaling_input, args.predictor_input, args.out)
    print(f"Wrote T51 revision artifacts: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
