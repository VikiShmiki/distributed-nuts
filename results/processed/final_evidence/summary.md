# Final mechanism-and-limits evidence summary

This summary includes only valid post-repair evidence used by `paper.tex`.

| evidence | value |
|---|---:|
| Final monolithic rows | 6 |
| Final bucketed rows | 24 |
| Bucketed rows faster than monolithic | 8 / 24 (33.3%) |
| Warm speedup, best / median / worst | 1.125 / 0.778 / 0.476 |
| Model families whose best row is faster | 3 / 6 |
| Final transition-metric mismatch rows | 0 / 24 |
| Focused history work ratio / speedup | 0.528 / 1.0425 |
| Primary gate | FAIL (`1.0425 < 1.05`) |
| Oracle-current work ratio / speedup | 0.347 / 1.4739 (analysis-only) |
| Valid GPU wall-clock rows | 0 |

Sources:

- `results/raw/broad_sweep_accept_targeted/*/summary.csv`
- `results/raw/oracle_gap/accept_targeted_cpu/summary.csv`
- `results/raw/correctness/gp_tiny_uturn_fixed/equivalence.json`
- rejected-control provenance:
  `/home/hpc/users/viktor.najdovski/abnuts_runs/{23078,23079}/raw/oracle_gap/`

Timing uses tree-wide blocking. Final CPU warm totals are the minimum of three
blocked repeats. Oracle-current is a non-deployable analysis upper bound. Run
`21070` and all other pre-T43 performance magnitudes are excluded.
