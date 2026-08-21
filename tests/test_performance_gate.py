from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

from abnuts.analysis.performance_gate import _evaluate_negative_controls


def test_performance_gate_cli_generates_fail_report(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_root = tmp_path / "raw21070"
    repair_root = tmp_path / "repair"
    correctness_dir = tmp_path / "correctness"
    oracle_dir = tmp_path / "oracle"
    processed_out = tmp_path / "processed"
    latex_out = tmp_path / "latex"

    _write_csv(
        raw_root / "chain_scaling" / "full" / "summary.csv",
        [
            {
                "method": "monolithic",
                "model": "funnel",
                "speedup_vs_monolithic": "1.0",
                "mono_total_leapfrog_count": "10",
                "bucket_total_leapfrog_count": "10",
            },
            {
                "method": "bucketed",
                "model": "funnel",
                "speedup_vs_monolithic": "0.95",
                "mono_total_leapfrog_count": "10",
                "bucket_total_leapfrog_count": "10",
            },
        ],
    )
    _write_csv(
        raw_root / "gaussian_process" / "full" / "summary.csv",
        [
            {
                "method": "bucketed",
                "model": "gaussian_process",
                "speedup_vs_monolithic": "0.80",
                "mono_total_leapfrog_count": "10",
                "bucket_total_leapfrog_count": "11",
            },
        ],
    )

    hetero_summary = repair_root / "summary.csv"
    _write_csv(
        hetero_summary,
        [
            {
                "method": "monolithic",
                "model": "funnel",
                "predictor": "monolithic",
                "bucket_size": "0",
                "num_chains": "8",
                "dimension": "4",
                "speedup": "1.0",
                "timing_warm_iteration_seconds": "0.10",
                "bucket_num_buckets": "",
                "bucket_padding_count": "0",
                "bucket_padding_ratio": "0.0",
            },
            {
                "method": "bucketed",
                "model": "funnel",
                "predictor": "history",
                "bucket_size": "2",
                "num_chains": "8",
                "dimension": "4",
                "speedup": "0.90",
                "timing_warm_iteration_seconds": "0.11",
                "bucket_num_buckets": "4",
                "bucket_padding_count": "0",
                "bucket_padding_ratio": "0.0",
            },
            {
                "method": "bucketed",
                "model": "funnel",
                "predictor": "history",
                "bucket_size": "4",
                "num_chains": "8",
                "dimension": "4",
                "speedup": "0.92",
                "timing_warm_iteration_seconds": "0.10",
                "bucket_num_buckets": "2",
                "bucket_padding_count": "0",
                "bucket_padding_ratio": "0.0",
            },
        ],
    )

    _write_csv(
        oracle_dir / "summary.csv",
        [
            {
                "scheduler_mode": "monolithic",
                "scheduler_label": "monolithic",
                "is_analysis_upper_bound": "False",
                "bucket_size": "0",
                "speedup": "1.0",
            },
            {
                "scheduler_mode": "history",
                "scheduler_label": "history",
                "is_analysis_upper_bound": "False",
                "bucket_size": "2",
                "speedup": "0.91",
            },
            {
                "scheduler_mode": "oracle_current",
                "scheduler_label": "oracle_current (analysis-only upper bound)",
                "is_analysis_upper_bound": "True",
                "bucket_size": "2",
                "speedup": "1.20",
            },
        ],
    )

    correctness_dir.mkdir(parents=True)
    (correctness_dir / "equivalence.json").write_text(
        json.dumps(
            {
                "equivalence_passed": True,
                "final_rng_keys_equal": True,
                "discrete_metrics_exact": True,
                "float_metrics_within_tolerance": True,
                "positions_allclose": True,
                "max_position_delta": 0.0,
                "metric_max_abs_delta": {"acceptance_statistic": 0.0},
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "abnuts.analysis.performance_gate",
            "--pre-repair-run-dir",
            str(raw_root),
            "--correctness-dir",
            str(correctness_dir),
            "--heterogeneous-summary",
            str(hetero_summary),
            "--oracle-gap-dir",
            str(oracle_dir),
            "--out",
            str(processed_out),
            "--latex-out",
            str(latex_out),
            "--overwrite",
        ],
        check=True,
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "Performance gate result: FAIL" in completed.stdout
    assert (processed_out / "gate_summary.csv").exists()
    assert (processed_out / "gate_report.md").exists()
    assert (processed_out / "manifest.json").exists()
    assert (latex_out / "gate_report.tex").exists()

    report = (processed_out / "gate_report.md").read_text(encoding="utf-8")
    assert "**Performance Gate: FAIL**" in report
    assert "Broad sweeps remain blocked" in report
    assert "Pre-repair run 21070 bucketed rows faster than monolithic: 0 / 2" in report
    assert "Best oracle-current analysis-only upper bound: 1.200x" in report

    rows = list(csv.DictReader((processed_out / "gate_summary.csv").open(encoding="utf-8")))
    status_by_gate = {row["gate"]: row["status"] for row in rows}
    assert status_by_gate["correctness"] == "PASS"
    assert status_by_gate["heterogeneous_speedup"] == "FAIL"
    assert status_by_gate["homogeneous_negative_control"] == "FAIL"
    assert status_by_gate["python_bucket_loop_overhead"] == "PASS"

    latex = (latex_out / "gate_report.tex").read_text(encoding="utf-8")
    assert "% Generated by: python -m abnuts.analysis.performance_gate" in latex
    assert "Performance Gate: FAIL" in latex


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_negative_control_uses_production_speedup_and_reports_diagnosis(tmp_path: Path) -> None:
    summary = tmp_path / "negative" / "summary.csv"
    _write_csv(
        summary,
        [
            {
                "method": "bucketed",
                "speedup_unprofiled_vs_monolithic": "0.86",
                "speedup_profiled_vs_monolithic": "0.72",
                "timing_breakdown_enabled": "True",
                "timing_component_jit_measurement_overhead_seconds": "0.03",
                "timing_unattributed_seconds": "0.05",
                "timing_repeated_planning_seconds": "0.04",
                "diagnostic_primary_overhead": "unattributed_fixed_cpu",
                "diagnostic_conclusion": "bounded production slowdown",
            }
        ],
    )

    passed, value, note, source = _evaluate_negative_controls((summary,), 0.80)

    assert passed is True
    assert value == "0.860x"
    assert "worst profiled speedup=0.720x" in note
    assert "component-measurement overhead" in note
    assert source == str(summary)
