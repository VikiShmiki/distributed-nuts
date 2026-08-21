# Performance Forensics Report — Run 21070

**Label:** NEGATIVE DIAGNOSTIC EVIDENCE from the pre-repair executor. Not final paper evidence.

**Run directory:** `/home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw`

## 1. Overall summary

| Metric | Value |
|--------|-------|
| Total monolithic rows | 48 |
| Total bucketed rows | 735 |
| Bucketed rows faster than monolithic | 21 / 735 (2.9%) |
| Best warm speedup | 1.2645x |
| Worst warm speedup | 0.0573x |
| Median warm speedup | 0.2439x |

## 2. Speedup by benchmark family

| Family | N (bucketed) | Faster | Best | Worst | Median |
|--------|-------------|--------|------|-------|--------|
| chain_scaling | 225 | 21 | 1.2645 | 0.0614 | 0.4911 |
| dimension_scaling | 225 | 0 | 0.3092 | 0.0573 | 0.1237 |
| eight_schools_centered | 45 | 0 | 0.7246 | 0.0720 | 0.2438 |
| eight_schools_noncentered | 45 | 0 | 0.5296 | 0.1198 | 0.2421 |
| funnel_ablation | 60 | 0 | 0.6320 | 0.0613 | 0.2009 |
| gaussian_process | 45 | 0 | 0.5120 | 0.0760 | 0.2486 |
| hierarchical_logistic | 45 | 0 | 0.4988 | 0.1204 | 0.2476 |
| stochastic_volatility | 45 | 0 | 0.4981 | 0.0673 | 0.2466 |

## 3. Speedup vs. number of buckets

This table shows whether slowdown scales with `num_buckets`.

| num_buckets | N | Faster | Median speedup | Best speedup |
|------------|---|--------|----------------|-------------|
| 1 | 90 | 21 | 0.9747 | 1.2645 |
| 2 | 135 | 0 | 0.4914 | 0.7246 |
| 4 | 210 | 0 | 0.2477 | 0.3634 |
| 8 | 195 | 0 | 0.1240 | 0.1841 |
| 16 | 105 | 0 | 0.0624 | 0.0786 |

**Finding:** Median speedup decreases monotonically as `num_buckets` increases. This is consistent with **Python per-bucket dispatch overhead dominating** warm runtime.

Approximate median speedup ratio between consecutive bucket-count doublings:
  - 1 → 2 buckets: median speedup changes by factor 0.504
  - 2 → 4 buckets: median speedup changes by factor 0.504
  - 4 → 8 buckets: median speedup changes by factor 0.501
  - 8 → 16 buckets: median speedup changes by factor 0.503

## 4. Gaussian-process leapfrog-count mismatches

**35 / 45 (77.8%) gaussian_process bucketed rows have `bucket_total_leapfrog_count != mono_total_leapfrog_count`.**

This is a correctness red flag. The leapfrog counts should match exactly under the same per-chain RNG sequence; any divergence suggests a bug in the bucketed scatter/gather or padding logic for this model.

Example mismatches (first 5):

| C | D | predictor | bucket_size | diff (bucket - mono) |
|---|---|-----------|-------------|----------------------|
| 512 | 3 | none | 64 | +27 |
| 512 | 3 | none | 128 | +7 |
| 512 | 3 | none | 256 | +26 |
| 512 | 3 | random | 64 | +27 |
| 512 | 3 | random | 128 | +7 |

## 5. Architecture diagnosis

The bucketed executor in `src/abnuts/nuts/bucketed.py` contains a Python `for` loop over `plan.num_buckets` (line 81 in the current source). Each loop iteration calls `monolithic_transition` — a separate JAX dispatch — for that bucket's chains.

Consequences:

- When `num_buckets=1` the loop is a single dispatch: bucketed can be faster than monolithic (best observed: **1.264x** for chain_scaling C=128, D=128).
- When `num_buckets>1` each extra bucket adds a synchronisation barrier and Python overhead. Observed median speedups: 1 buckets → 0.975x, 2 buckets → 0.491x, 4 buckets → 0.248x, 8 buckets → 0.124x, 16 buckets → 0.062x.
- Slowdown appears to scale roughly **linearly with `num_buckets`**, consistent with serial Python dispatch dominating the warm hot path.

This architecture does not match the intended design (host-side planning + device-side fixed-shape/single-JIT executor). Repair is required before broad sweeps.

## 6. What was not available

- Per-bucket timing breakdown was not enabled in run 21070 (`timing_breakdown_enabled: false` in all manifests). Gather / executor / scatter split is not available from this run.
- No profiler marker data is available from run 21070.

## 7. Next steps

- **T33:** JIT baseline parity — jitted monolithic vs. current monolithic.
- **T34:** Real fixed-shape bucket executor — eliminate Python dispatch loop.
- **T35:** Correctness regression gate — confirm GP leapfrog counts match.

Full paper sweeps are **blocked** until the performance gate (T37) passes.
