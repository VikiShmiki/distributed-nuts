# Performance Gate Before Paper Sweeps

**Performance Gate: PASS**

This report is generated from the raw artifacts named under Sources below. It does not run a broad sweep.

## Decision

Broad sweeps may proceed to the minimal HPC validation task.

Next task guidance: Promote T38 minimal HPC validation.

## Gate Criteria

| Gate | Status | Threshold | Value | Evidence |
|------|--------|-----------|-------|----------|
| correctness | PASS | equivalence_passed and exact discrete/RNG metrics | equivalence_passed=True; final_rng_keys_equal=True; discrete_metrics_exact=True; float_metrics_within_tolerance=True; positions_allclose=True | max_position_delta=0.0; metric_max_abs_delta={'acceptance_statistic': 0.0, 'energy_error': 0.0, 'gradient_norm': 0.0} |
| mechanism_executed_work | PASS | non-analysis executed-work ratio < 1.00 (hardware independent) | 0.607 | Best non-analysis executed-work ratio 0.607 from oracle-gap history, bucket_size=64; oracle plans on the same realized depths reach 0.380; mean predictor abs error 3.457; planner is 4.1% of warm time; oracle_current wall speedup 1.833x (analysis-only). |
| heterogeneous_speedup | PASS | non-analysis repaired speedup >= 1.05x | 1.286x | Best non-analysis repaired row: oracle-gap history, bucket_size=128 |
| homogeneous_negative_control | PASS | homogeneous control present, slowdown explained, speedup >= 0.80x | 0.859x | worst production speedup=0.859x; best production speedup=0.859x; worst profiled speedup=0.748x; best profiled speedup=0.748x; max component-measurement overhead=0.042368s; max unattributed=0.063868s; max repeated planning=0.038643s; primary overhead=executor; timing breakdown and diagnostic explanation present for all bucketed rows=True |
| python_bucket_loop_overhead | PASS | repaired warm runtime max/min ratio <= 1.25 | 1.162 | No-padding repaired warm medians by bucket count: 1 buckets=0.190039s, 2 buckets=0.197377s, 4 buckets=0.220839s |
| report_from_raw_results | PASS | all named raw summaries are present | present | Gate uses repaired raw summaries plus T35 equivalence JSON; T32 pre-repair source: /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw. |
| faster_rows_reported | PASS | report includes bucketed faster-row counts | pre-repair 21/735; repaired 4/11 | Counts are computed directly from speedup columns. |

## Faster Rows

Pre-repair run 21070 bucketed rows faster than monolithic: 21 / 735.
Repaired local non-analysis bucketed rows faster than monolithic: 4 / 11.
Pre-repair speedups are negative diagnostic evidence only: best 1.264x, median 0.244x.
Best repaired non-analysis candidate: 1.286x (oracle-gap history, bucket_size=128).
Best oracle-current analysis-only upper bound: 1.833x (oracle-gap oracle_current (analysis-only upper bound), bucket_size=64 analysis-only).

## Mechanism Versus Scheduler

Executed lane-steps are hardware independent, so they separate three things wall-clock timing mixes together: whether the executor reclaims straggler work, whether the scheduler finds that work, and what planning costs.

| quantity | value | reading |
|---|---|---|
| deployed scheduler executed-work ratio | 0.607 | what oracle-gap history, bucket_size=64 actually reclaimed |
| oracle-plan executed-work ratio | 0.380 | what the same executor reclaims given perfect grouping |
| mean predictor absolute error | 3.457 | realized-depth prediction error driving the gap above |
| planner share of warm time | 4.1% | scheduling cost paid out of the reclaimed budget |
| oracle_current wall speedup | 1.833x | analysis-only upper bound, cannot satisfy the wall-clock gate |

The executor reclaims work when given good plans, and the deployed predictor does not supply them: the gap between the two ratios above is 0.227. The remaining loss is scheduler quality, not executor structure.

## Correctness And Diagnostics

Run 21070 GP leapfrog-count mismatches remain historical pre-repair diagnostic evidence: 35 / 45 GP bucketed rows.
The repaired GP correctness gate is evaluated from T35 equivalence JSON.

## Bucket-Count Scaling

Repaired no-padding warm runtime max/min ratio across bucket-count rows: 1.162.

## Sources

- correctness: `results/raw/correctness/gp_tiny_uturn_fixed/equivalence.json`
- mechanism_executed_work: `results/raw/oracle_gap/uturn_fixed_repeats/summary.csv`
- heterogeneous_speedup: `results/raw/t44_repeat_timing/tiny_cpu/summary.csv, results/raw/t44_repeat_timing/tiny_cpu_bucket_scaling/summary.csv; results/raw/oracle_gap/uturn_fixed_repeats/summary.csv`
- homogeneous_negative_control: `results/raw/performance_gate/homogeneous_negative_control_uturn_fixed/summary.csv`
- python_bucket_loop_overhead: `results/raw/t44_repeat_timing/tiny_cpu/summary.csv, results/raw/t44_repeat_timing/tiny_cpu_bucket_scaling/summary.csv`
- report_from_raw_results: `/home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw, results/raw/correctness/gp_tiny_uturn_fixed/equivalence.json, results/raw/t44_repeat_timing/tiny_cpu/summary.csv, results/raw/t44_repeat_timing/tiny_cpu_bucket_scaling/summary.csv, results/raw/oracle_gap/uturn_fixed_repeats/summary.csv, results/raw/performance_gate/homogeneous_negative_control_uturn_fixed/summary.csv`
- faster_rows_reported: `/home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw; repaired summaries`
