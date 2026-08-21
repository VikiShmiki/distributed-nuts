# RESEARCH.md — Final Research Framing

## Final claim

This project is complete as a **mechanism-and-limits study** of work-aware
multi-chain NUTS scheduling. It does not establish a general GPU or accelerator
speedup.

The final defensible claim is:

> Fixed-shape bucket scheduling can preserve the tested NUTS transition and
> reduce maximum-driven lane work when per-chain trajectory length is
> heterogeneous. A history-based online scheduler converts that mechanism into
> modest CPU gains on some targets, but prediction error and fixed scheduling
> costs prevent consistent end-to-end improvement after step sizes are tuned to
> the usual NUTS acceptance target.

## Valid final evidence

Only post-correctness-repair evidence with independently selected step sizes is
used for headline performance results.

- `results/raw/t51_preferred_v2/final_suite/`: six CPU models, 512 chains, 16
  transitions, five sampler seeds, an explicit cold priming call, and 30
  retained blocked warm repetitions, with bucket widths 64 and 128 and
  unsorted, history, and causal last-depth plans. Of 36 aggregated bucketed
  configurations, 6 (16.7%) have median speedup above one. Median, best, and
  worst speedups are `0.896x`, `1.059x`, and `0.529x`. The best row is faster
  for 3 of 6 model families. Aggregate leapfrog counts match monolithic in all
  180 seed/configuration rows.
- `results/raw/t51_preferred_v2/scaling/`: controlled funnel and Gaussian-process
  chain scaling at `C={128,256,512,1024}`, with the same five-seed/30-repeat
  protocol. Best 128-chain rows are slow (`0.963x` and `0.929x`); at 1024
  chains the best rows reach `1.312x` (`95% CI [1.227,1.434]`) and `1.136x`
  (`[1.027,1.173]`). This supports an amortization boundary on CPU, not a
  general accelerator claim.
- `results/raw/t51_preferred/predictor_diagnostics/`: five-seed per-chain funnel
  predictions. Last-depth improves MAE (`1.288 -> 0.307`) and deepest-decile
  recall (`0.492 -> 0.672`) over history but does not consistently improve
  fixed-width bucket maxima or wall time.
- `results/raw/oracle_gap/accept_targeted_cpu/`: focused funnel mechanism and
  oracle decomposition. History at width 64 reaches work ratio `0.528` and
  speedup `1.0425x`, below the pre-specified `1.05x` primary gate.
  Oracle-current reaches work ratio `0.347` and `1.4739x`, but is an
  analysis-only upper bound.
- The current correctness suite and
  `results/raw/correctness/gp_tiny_uturn_fixed/` establish shared-key
  monolithic/bucketed parity, including the Gaussian-process family and padded
  lane noninterference. Discrete metrics and RNG keys are exact; floating
  metrics use the documented `1e-5` tolerance.

## Evidence that is not a final performance result

- Run `21070` is pre-repair negative diagnostic evidence. It had `21/735`
  bucketed rows faster than monolithic, median speedup `0.244x`, and leapfrog
  mismatches in `35/45` Gaussian-process rows. Its transition was fully
  unrolled and later found to share a defective U-turn rule, so none of its
  performance magnitudes supports the final claim.
- All pre-T43 performance and posterior-quality outputs are void. The backward
  U-turn check made depth depend on integration direction, and consistency
  between wrappers could not detect the shared defect.
- Post-repair funnel results at step size `0.03` are sensitivity evidence only.
  Mean acceptance was `0.997`; ordinary step-size adaptation would avoid much
  of the work bucketing appeared to reclaim.
- GPU runs `23078` and `23079` are rejected. In run `23079`, one-bucket history
  and oracle-current execute the same `974,848` lane-steps but differ by
  `1.78x` in time. Every GPU wall-clock number remains invalid, including the
  apparently favorable `1.301x` row.

## Mechanism and binding limits

The repaired transition uses data-dependent `lax.while_loop` control flow. The
bucket executor uses compiled `lax.map`, giving each bucket an independent
batched stopping condition without a Python loop or dispatch per bucket.
Flattening or vectorizing the bucket axis would restore a global stopping
condition and erase the mechanism.

The executed-lane-work proxy is

```text
monolithic = C * max(chain leapfrog work)
bucketed   = sum(bucket width * max(work within bucket))
```

It is a hardware-independent proxy, not an instruction counter. Work reduction
is necessary but insufficient: the host must predict and sort chains, move a
fixed rectangle, execute narrower sequential buckets, and scatter valid lanes.

The focused accepted-target result identifies the binding limits:

1. history leaves work ratio `0.528` where current-depth oracle grouping reaches
   `0.347`;
2. history mean absolute depth error is `1.284`;
3. planning, gather, scatter, and residual overhead consume most of the saved
   executor budget; and
4. acceptance calibration halves the monolithic workload relative to the
   under-tuned configuration.

## Methodological scope

The final CPU suite is a short systems microbenchmark, not a posterior-quality
study. It uses five sampler seeds and 30 retained blocked warm repetitions, but
still uses small models, 16 transitions, coarse offline step-size calibration,
and no mass-matrix adaptation. The centered Eight Schools and Gaussian-process
configurations retain divergences. No long-chain ESS, convergence, or valid
accelerator result is claimed.

T51-v2 headline timings use the production outer path, an explicit unreported
cold priming call, and hierarchical bootstrap intervals over seeds and
independent method timing samples. The initial `results/raw/t51_preferred/`
pass is excluded because its first retained observation included compilation. Older component
timers remain diagnostic rather than exact additive accounting and are not
combined with the T51 intervals.

## Final research conclusion

The original approximately `1.28x` A100 target is not reproduced. The study's
contribution is instead a correctness-preserving mechanism, predictor-tail
diagnostics, and a controlled CPU amortization boundary: both scaling targets
are slower at 128 chains and faster at 1024 chains under the preferred
five-seed/30-repeat protocol. Bucketing helps only when trajectory work is
predictable and large enough to amortize the scheduler. The unresolved GPU
question is a limitation, not a pending claim.
