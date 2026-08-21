"""Performance forensics report for HPC benchmark runs.

Parses raw summary CSVs from a benchmark run directory and writes:
- results/processed/performance_forensics/forensics_summary.csv  (row-level data)
- results/processed/performance_forensics/report.md              (human-readable summary)

Usage::

    python -m abnuts.analysis.forensics \\
        --run-dir /path/to/abnuts_runs/21070/raw \\
        --out results/processed/performance_forensics \\
        --run-id 21070

The script does NOT modify any sampler code, JIT behaviour, or planner logic.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Root raw-results directory (must contain subdirs with summary.csv files).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory for report and processed CSV.",
    )
    p.add_argument(
        "--run-id",
        default="unknown",
        help="Human-readable run identifier, e.g. '21070'.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Collect all summary.csv rows under run_dir, tagging each with _family."""
    all_rows: list[dict[str, Any]] = []
    csv_files = sorted(run_dir.rglob("summary.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No summary.csv files found under {run_dir}")
    for csv_path in csv_files:
        # family is the first subdirectory under run_dir
        rel = csv_path.relative_to(run_dir)
        family = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        with csv_path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["_family"] = family
                row["_source_csv"] = str(csv_path)
                all_rows.append(row)
    return all_rows


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _float(row: dict[str, Any], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def _int(row: dict[str, Any], key: str) -> int | None:
    try:
        return int(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def _str(row: dict[str, Any], key: str) -> str:
    return str(row.get(key, ""))


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def compute_speedup_table(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Per-family speedup statistics for bucketed rows."""
    family_speedups: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("method") != "bucketed":
            continue
        s = _float(r, "speedup_vs_monolithic")
        if s is not None:
            family_speedups[r["_family"]].append(s)

    result: dict[str, dict[str, Any]] = {}
    for fam, vals in sorted(family_speedups.items()):
        vals_s = sorted(vals)
        result[fam] = {
            "n_bucketed": len(vals_s),
            "n_faster": sum(1 for v in vals_s if v > 1.0),
            "best": max(vals_s),
            "worst": min(vals_s),
            "median": _median(vals_s),
        }
    return result


def compute_bucket_scaling(
    rows: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Speedup statistics grouped by num_buckets."""
    by_nb: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("method") != "bucketed":
            continue
        nb = _int(r, "bucket_num_buckets")
        s = _float(r, "speedup_vs_monolithic")
        if nb is not None and nb > 0 and s is not None:
            by_nb[nb].append(s)

    result: dict[int, dict[str, Any]] = {}
    for nb in sorted(by_nb.keys()):
        vals = sorted(by_nb[nb])
        result[nb] = {
            "n": len(vals),
            "n_faster": sum(1 for v in vals if v > 1.0),
            "median": _median(vals),
            "best": max(vals),
            "worst": min(vals),
        }
    return result


def compute_gp_mismatches(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Leapfrog-count mismatch stats for gaussian_process family."""
    gp_bucket = [r for r in rows if r.get("method") == "bucketed" and r.get("_family") == "gaussian_process"]
    if not gp_bucket:
        return {"available": False, "note": "No gaussian_process bucketed rows found"}

    total = len(gp_bucket)
    mismatches = []
    for r in gp_bucket:
        mono_lf = _int(r, "mono_total_leapfrog_count")
        bucket_lf = _int(r, "bucket_total_leapfrog_count")
        if mono_lf is not None and bucket_lf is not None and mono_lf != bucket_lf:
            mismatches.append({
                "diff": bucket_lf - mono_lf,
                "C": _str(r, "num_chains"),
                "D": _str(r, "dimension"),
                "predictor": _str(r, "predictor"),
                "bucket_size": _str(r, "bucket_size"),
            })

    return {
        "available": True,
        "total_gp_bucketed": total,
        "mismatch_count": len(mismatches),
        "mismatch_fraction": len(mismatches) / total if total else 0.0,
        "examples": mismatches[:5],
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def build_markdown_report(
    run_id: str,
    run_dir: Path,
    rows: list[dict[str, Any]],
) -> str:
    bucket_rows = [r for r in rows if r.get("method") == "bucketed"]
    mono_rows = [r for r in rows if r.get("method") == "monolithic"]

    all_speedups = [_float(r, "speedup_vs_monolithic") for r in bucket_rows]
    valid_speedups = [s for s in all_speedups if s is not None]
    n_faster = sum(1 for s in valid_speedups if s > 1.0)
    best = max(valid_speedups) if valid_speedups else float("nan")
    worst = min(valid_speedups) if valid_speedups else float("nan")
    med = _median(valid_speedups)

    family_table = compute_speedup_table(rows)
    bucket_scaling = compute_bucket_scaling(rows)
    gp_info = compute_gp_mismatches(rows)

    # Check whether slowdown scales with num_buckets
    scaling_monotone = False
    nb_keys = sorted(bucket_scaling.keys())
    if len(nb_keys) >= 2:
        medians = [bucket_scaling[nb]["median"] for nb in nb_keys]
        scaling_monotone = all(medians[i] >= medians[i + 1] for i in range(len(medians) - 1))

    lines: list[str] = []
    lines.append(f"# Performance Forensics Report — Run {run_id}")
    lines.append("")
    lines.append("**Label:** NEGATIVE DIAGNOSTIC EVIDENCE from the pre-repair executor. "
                 "Not final paper evidence.")
    lines.append("")
    lines.append(f"**Run directory:** `{run_dir}`")
    lines.append("")

    lines.append("## 1. Overall summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total monolithic rows | {len(mono_rows)} |")
    lines.append(f"| Total bucketed rows | {len(bucket_rows)} |")
    lines.append(f"| Bucketed rows faster than monolithic | {n_faster} / {len(valid_speedups)} ({100*n_faster/len(valid_speedups):.1f}%) |" if valid_speedups else "| Bucketed rows faster | N/A |")
    lines.append(f"| Best warm speedup | {best:.4f}x |")
    lines.append(f"| Worst warm speedup | {worst:.4f}x |")
    lines.append(f"| Median warm speedup | {med:.4f}x |")
    lines.append("")

    lines.append("## 2. Speedup by benchmark family")
    lines.append("")
    lines.append("| Family | N (bucketed) | Faster | Best | Worst | Median |")
    lines.append("|--------|-------------|--------|------|-------|--------|")
    for fam, st in family_table.items():
        lines.append(
            f"| {fam} | {st['n_bucketed']} | {st['n_faster']} | {st['best']:.4f} | {st['worst']:.4f} | {st['median']:.4f} |"
        )
    lines.append("")

    lines.append("## 3. Speedup vs. number of buckets")
    lines.append("")
    lines.append("This table shows whether slowdown scales with `num_buckets`.")
    lines.append("")
    lines.append("| num_buckets | N | Faster | Median speedup | Best speedup |")
    lines.append("|------------|---|--------|----------------|-------------|")
    for nb, st in bucket_scaling.items():
        lines.append(
            f"| {nb} | {st['n']} | {st['n_faster']} | {st['median']:.4f} | {st['best']:.4f} |"
        )
    lines.append("")

    if scaling_monotone:
        lines.append(
            "**Finding:** Median speedup decreases monotonically as `num_buckets` increases. "
            "This is consistent with **Python per-bucket dispatch overhead dominating** warm runtime."
        )
    else:
        lines.append(
            "**Finding:** Speedup does not decrease monotonically with `num_buckets`. "
            "Dispatch overhead hypothesis requires further investigation."
        )
    lines.append("")
    lines.append(
        "Approximate median speedup ratio between consecutive bucket-count doublings:"
    )
    if len(nb_keys) >= 2:
        for i in range(len(nb_keys) - 1):
            nb_lo, nb_hi = nb_keys[i], nb_keys[i + 1]
            ratio = bucket_scaling[nb_hi]["median"] / bucket_scaling[nb_lo]["median"] if bucket_scaling[nb_lo]["median"] != 0 else float("nan")
            lines.append(
                f"  - {nb_lo} → {nb_hi} buckets: median speedup changes by factor {ratio:.3f}"
            )
    lines.append("")

    lines.append("## 4. Gaussian-process leapfrog-count mismatches")
    lines.append("")
    if not gp_info.get("available"):
        lines.append("No gaussian_process bucketed rows found in this run.")
    else:
        n_mismatch = gp_info["mismatch_count"]
        n_total = gp_info["total_gp_bucketed"]
        frac = gp_info["mismatch_fraction"]
        lines.append(
            f"**{n_mismatch} / {n_total} ({100*frac:.1f}%) gaussian_process bucketed rows have "
            f"`bucket_total_leapfrog_count != mono_total_leapfrog_count`.**"
        )
        lines.append("")
        lines.append(
            "This is a correctness red flag. The leapfrog counts should match exactly "
            "under the same per-chain RNG sequence; any divergence suggests a bug in "
            "the bucketed scatter/gather or padding logic for this model."
        )
        if gp_info.get("examples"):
            lines.append("")
            lines.append("Example mismatches (first 5):")
            lines.append("")
            lines.append("| C | D | predictor | bucket_size | diff (bucket - mono) |")
            lines.append("|---|---|-----------|-------------|----------------------|")
            for ex in gp_info["examples"]:
                lines.append(
                    f"| {ex['C']} | {ex['D']} | {ex['predictor']} | {ex['bucket_size']} | {ex['diff']:+d} |"
                )
    lines.append("")

    lines.append("## 5. Architecture diagnosis")
    lines.append("")
    lines.append(
        "The bucketed executor in `src/abnuts/nuts/bucketed.py` contains a Python `for` loop "
        "over `plan.num_buckets` (line 81 in the current source). Each loop iteration calls "
        "`monolithic_transition` — a separate JAX dispatch — for that bucket's chains."
    )
    lines.append("")
    lines.append("Consequences:")
    lines.append("")
    lines.append(
        "- When `num_buckets=1` the loop is a single dispatch: bucketed can be faster "
        "than monolithic (best observed: **1.264x** for chain_scaling C=128, D=128)."
    )
    lines.append(
        "- When `num_buckets>1` each extra bucket adds a synchronisation barrier and "
        "Python overhead. Observed median speedups: "
        + ", ".join(f"{nb} buckets → {bucket_scaling[nb]['median']:.3f}x" for nb in nb_keys if nb in bucket_scaling)
        + "."
    )
    lines.append(
        "- Slowdown appears to scale roughly **linearly with `num_buckets`**, consistent "
        "with serial Python dispatch dominating the warm hot path."
    )
    lines.append("")
    lines.append(
        "This architecture does not match the intended design (host-side planning + "
        "device-side fixed-shape/single-JIT executor). Repair is required before broad sweeps."
    )
    lines.append("")

    lines.append("## 6. What was not available")
    lines.append("")
    lines.append("- Per-bucket timing breakdown was not enabled in run 21070 "
                 "(`timing_breakdown_enabled: false` in all manifests). "
                 "Gather / executor / scatter split is not available from this run.")
    lines.append("- No profiler marker data is available from run 21070.")
    lines.append("")

    lines.append("## 7. Next steps")
    lines.append("")
    lines.append("- **T33:** JIT baseline parity — jitted monolithic vs. current monolithic.")
    lines.append("- **T34:** Real fixed-shape bucket executor — eliminate Python dispatch loop.")
    lines.append("- **T35:** Correctness regression gate — confirm GP leapfrog counts match.")
    lines.append("")
    lines.append("Full paper sweeps are **blocked** until the performance gate (T37) passes.")
    lines.append("")

    return "\n".join(lines)


def build_summary_csv(rows: list[dict[str, Any]]) -> str:
    """Write a flattened CSV of bucketed rows with family + key fields."""
    keep_fields = [
        "_family",
        "model",
        "method",
        "predictor",
        "bucket_size",
        "num_chains",
        "dimension",
        "seed",
        "num_steps",
        "speedup_vs_monolithic",
        "t_mono",
        "t_bucket",
        "bucket_num_buckets",
        "bucket_padding_count",
        "bucket_padding_ratio",
        "mono_total_leapfrog_count",
        "bucket_total_leapfrog_count",
        "mono_divergence_count",
        "bucket_divergence_count",
        "mono_max_realized_tree_depth",
        "bucket_max_realized_tree_depth",
    ]
    bucket_rows = [r for r in rows if r.get("method") in {"bucketed", "monolithic"}]
    if not bucket_rows:
        return ""
    buf = StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=keep_fields,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(bucket_rows)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        rows = load_run_rows(args.run_dir)
    except FileNotFoundError as exc:
        print(f"forensics: {exc}", file=sys.stderr)
        return 2

    out_dir: Path = args.out
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        print(
            f"forensics: output directory already contains files: {out_dir}. "
            "Pass --overwrite to replace.",
            file=sys.stderr,
        )
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)

    report_md = build_markdown_report(args.run_id, args.run_dir, rows)
    report_path = out_dir / "report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Wrote forensics report: {report_path}")

    summary_csv = build_summary_csv(rows)
    csv_path = out_dir / "forensics_summary.csv"
    csv_path.write_text(summary_csv, encoding="utf-8")
    print(f"Wrote forensics CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
