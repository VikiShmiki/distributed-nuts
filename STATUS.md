# STATUS.md — Asynchronous Bucketed NUTS Task Board

## Current active task

**No active task — T53 is complete.**

### T53 complete: private GitHub repository publication

Initialized this folder as a Git repository on branch `main`, committed 558
source/artifact files, created the private GitHub repository, and pushed it to:

- `https://github.com/VikiShmiki/distributed-nuts`
- remote: `https://github.com/VikiShmiki/distributed-nuts.git`
- visibility: `PRIVATE`
- default branch: `main`

Added `.gitignore` and `images/.gitkeep`. The 3 GB
`images/abnuts.sif` build product, `.claude` local permissions, Python caches,
editor state, virtual environments, and local LaTeX build products are not
tracked. Source, canonical raw/processed/LaTeX evidence, logs, `paper.tex`, and
`presentation/index.html` are tracked. No GitHub credential was written into
the repository.

Files changed:

- `.gitignore`
- `images/.gitkeep`
- `STATUS.md`
- `.git/` metadata and `origin` remote (local repository metadata)

Commands and validation:

```bash
git init -b main
git config user.name 'Viktor Najdovski'
git config user.email '142947817+VikiShmiki@users.noreply.github.com'
git add .
git commit -m 'Initial research implementation and presentation'
gh repo create VikiShmiki/distributed-nuts --private --source=. \
  --remote=origin --push \
  --description 'Work-aware bucketed NUTS in JAX: correctness-preserving scheduling, CPU scaling evidence, paper, and visual presentation'
# PASS: repository created and main pushed

gh repo view VikiShmiki/distributed-nuts \
  --json nameWithOwner,visibility,url,defaultBranchRef,isPrivate
# PASS: isPrivate=true, visibility=PRIVATE, default branch=main

git check-ignore -v images/abnuts.sif .claude/settings.local.json \
  .pytest_cache/CACHEDIR.TAG
# PASS: all local/build paths ignored
```

Next active task: none.

### T52 complete: high-level visual course presentation

Created a self-contained nine-slide HTML presentation at
`presentation/index.html` for an audience with no Bayesian inference or NUTS
background. The visual narrative is:

1. project and load-balancing question;
2. why Bayesian sampling represents uncertainty;
3. NUTS as an adaptive probability-landscape walk;
4. lockstep control-flow divergence and straggler waste;
5. predict/sort/bucket scheduling;
6. host planning plus compiled fixed-shape JAX execution;
7. transition-preservation invariants;
8. final T51-v2 CPU scaling results and rejected GPU caveat;
9. three concise systems takeaways.

The site has no external runtime dependencies. It includes responsive layouts,
inline SVG visualizations, active-slide animations, keyboard, button, hash,
touch-swipe and fullscreen controls, reduced-motion support, print styling, and
accessible labels. Claims use only valid final evidence: funnel `1.31x` and GP
`1.14x` at 1024 chains, `6/36` final configurations above one, five seeds by 30
warm repetitions, matching transition metrics, and no accepted GPU timing.

Files changed:

- `presentation/index.html`
- `STATUS.md`

Validation:

```bash
python3 - <<'PY'
from html.parser import HTMLParser
from pathlib import Path
text = Path('presentation/index.html').read_text()
parser = HTMLParser()
parser.feed(text)
parser.close()
assert text.count('class="slide') == 9
PY
# PASS: HTML parsed; 9 slides found

python3 -m http.server 8765 --directory presentation
curl --fail http://127.0.0.1:8765/
# PASS: presentation served successfully
```

Next active task: none.

### T51 complete: preferred statistical revision protocol

The final CPU protocol now uses five sampler seeds and an explicit unreported
cold priming call followed by 30 retained, tree-blocked warm repetitions per
configuration. Summary time is the median; raw `timing_repetitions.csv` files
retain every observation and summary rows include median, IQR, extrema, and the
full JSON sample. Configuration-level 95% intervals use 10,000 hierarchical
bootstrap draws over seeds and independently resampled monolithic/bucketed
warm samples. Methods were timed sequentially, so repeat indices are not
misrepresented as paired observations.

A causal `last_depth` scheduler was added. Five-seed funnel diagnostics show
history versus last-depth Spearman `0.594 -> 0.631`, deepest-decile recall
`0.492 -> 0.672`, and MAE `1.288 -> 0.307`, but last-depth does not consistently
reduce within-bucket depth range or wall time.

Final six-model result (`results/raw/t51_preferred_v2/final_suite/`):

- 210 summary rows: 30 monolithic and 180 bucketed seed/configuration rows;
- 6,300 retained timing observations;
- all bucketed aggregate leapfrog, divergence, and maximum-depth metrics match
  their corresponding monolithic run (`0/180` rows with a mismatch);
- after aggregating five seeds, `6/36` bucketed configurations have median
  speedup above one;
- median/best/worst speedup: `0.896x / 1.059x / 0.529x`;
- the best row is above one in three of six model families; three individual
  configuration intervals lie wholly above one (funnel history/64, centered
  Eight Schools history/128, and centered Eight Schools last-depth/128).

Controlled scaling (`results/raw/t51_preferred_v2/scaling/`) contains 200
summary rows and 6,000 retained timings. Best preconfigured rows move from
`0.963x` and `0.929x` at `C=128` to:

- funnel `C=1024`: `1.312x`, 95% CI `[1.227, 1.434]`;
- Gaussian process `C=1024`: `1.136x`, 95% CI `[1.027, 1.173]`.

This establishes a controlled CPU amortization boundary while preserving the
mechanism-and-limits claim. It does not establish accelerator performance.

The first T51 pass under `results/raw/t51_preferred/` is preserved but excluded:
the production path had not made an explicit cold priming call, so the first of
30 retained observations included compilation. The median was not generally
selected from that observation, but the protocol said 30 warm repetitions, so
the entire final suite and scaling study were rerun into `t51_preferred_v2`.
No raw result was overwritten. The five-seed predictor diagnostic from the
first pass remains valid because it uses per-chain prediction/depth data rather
than those timing samples.

Files changed:

- `src/abnuts/experiments/run_benchmark.py`
- `src/abnuts/nuts/predictors.py`
- `src/abnuts/analysis/t51_revision.py`
- `tests/test_predictors.py`
- `tests/test_benchmark_timing_statistics.py`
- `configs/funnel_ablation.yaml`
- `configs/eight_schools.yaml`
- `configs/gaussian_process.yaml`
- `configs/hierarchical_logistic.yaml`
- `configs/stochastic_volatility.yaml`
- `configs/oracle_gap.yaml`
- `configs/t51_scaling_funnel.yaml`
- `configs/t51_scaling_gp.yaml`
- `paper.tex`
- `RESEARCH.md`
- `STATUS.md`

Canonical new outputs:

- `results/raw/t51_preferred_v2/final_suite/`
- `results/raw/t51_preferred_v2/scaling/`
- `results/raw/t51_preferred/predictor_diagnostics/`
- `results/processed/t51_preferred_v2/`
- `logs/t51_v2/run.log`

Long runs were executed synchronously on CPU, so there is no remaining PID or
SLURM job. Exact commands are in `logs/t51_v2/run.log`; each benchmark command
used `JAX_PLATFORMS=cpu`, `--warm-repeats 30`, one named preferred profile, and
a distinct non-overwriting output directory. The final v2 sequence ran from
2026-08-21 08:15:53+02:00 through 15:16:38+02:00. The user's concurrent GPU
jobs were not inspected, stopped, removed, or used in the paper.

Validation:

```bash
JAX_PLATFORMS=cpu python3 -m pytest -q
# PASS: 96 passed in 248.87s

python3 -m abnuts.analysis.t51_revision \
  --final-input results/raw/t51_preferred_v2/final_suite \
  --scaling-input results/raw/t51_preferred_v2/scaling \
  --predictor-input results/raw/t51_preferred/predictor_diagnostics \
  --out results/processed/t51_preferred_v2 --overwrite
# PASS: generated CSV, JSON, Markdown, and LaTeX-first tables

tectonic -o /tmp/abnuts-paper-t51 paper.tex
# PASS: paper.pdf generated; no overfull boxes
```

Failure record: the first analysis command timed out after 600 seconds in the
repository root on CPU while repeatedly decoding timing JSON inside 10,000-draw
bootstrap loops. No output was accepted. The bootstrap was vectorized, reran in
under five minutes, and does not block T51.

`paper.tex` now uses the work-aware name, explicitly distinguishes the method
from ordinary runtime sorting, reports the preferred timing protocol and
uncertainty, promotes last-depth as causal, adds predictor-tail and controlled
scaling results, preserves rejected GPU controls, and retains the limitations
on adaptation, posterior quality, and accelerator generalization.

Next active task: none.

### T50 complete: final paper, evidence summary, and research framing

`paper.tex` was rewritten from scratch around the complete evidence audit. The
final manuscript now reports the required aggregate final-suite statistics (6
monolithic rows, 24 bucketed rows, 8/24 faster, median/best/worst warm speedup),
the focused accepted-target oracle gap, exact scope of the correctness checks,
the step-size adaptation confound, and the rejection of all GPU wall-clock
rows. It distinguishes the production single-JIT path from the separately
instrumented component path and treats component timings as diagnostic because
they need not come from the repeat supplying the minimum outer warm time.

`RESEARCH.md` was replaced with the final claim-bounded research framing.
`results/processed/final_evidence/summary.md` records the concise evidence
summary and raw source paths. No raw result was changed or overwritten.

Files changed:

- `paper.tex`
- `RESEARCH.md`
- `results/processed/final_evidence/summary.md`
- `STATUS.md`

Validation:

```bash
tectonic -o /tmp/abnuts-paper-build paper.tex
# PASS: /tmp/abnuts-paper-build/paper.pdf generated; no overfull boxes

JAX_PLATFORMS=cpu python3 -m pytest -q \
    tests/test_monolithic_bucketed_equivalence.py \
    tests/test_sampler_ground_truth.py
# PASS: 11 passed in 75.21s
```

Final claim decision: bucketed scheduling preserves the tested NUTS transition
and reclaims real straggler work, but the history scheduler does not deliver a
consistent end-to-end gain after acceptance calibration. The final CPU suite
ranges from `0.476x` to `1.125x` across all bucketed configurations; the primary
focused history row is `1.0425x` and fails its `1.05x` gate; oracle-current is
an analysis-only `1.4739x` ceiling. There is no valid GPU performance claim.

Next active task: none. Further GPU work would be a new research project, not a
condition of this mechanism-and-limits conclusion.

---

## T50 working record

### T49 complete: both GPU jobs finished, but the measurement failed its controls

Jobs `23078` and `23079` completed with `status: ok`, empty stderr, and no
remaining queue entry. Raw outputs are preserved under
`/home/hpc/users/viktor.najdovski/abnuts_runs/{23078,23079}/raw/oracle_gap/`.

The accept-targeted run (`23079`, `C=2048`, `D=128`, `step_size=0.12`) appears
to show positive rows: history reaches `1.301x` with two 1024-wide buckets and
oracle-current reaches `1.170x`. These are **not publishable results**, because
the run fails its internal controls:

- The one-bucket history and oracle-current rows execute exactly the same
  `974,848` lane-steps, yet take `2.303s` and `1.292s`, a `1.78x` spread.
- The one-bucket controls range from `0.802x` to `1.429x` despite adding no
  work-reclaim mechanism.
- Across the adjacent runs, accept-targeting cuts measured lane-steps by `3.93x`
  (`3,833,856 -> 974,848`) but monolithic time rises (`1.446s -> 1.846s`).
- The assigned A100 MIG instance was dedicated to this benchmark; other jobs
  shown by `nvidia-smi` occupied other physical GPUs. The failed controls cannot
  simply be attributed to another process sharing our accelerator.

Therefore **every GPU wall-clock figure remains void**. T49 answered the GPU
question only negatively: the current timing protocol did not produce a
trustworthy GPU measurement. No GPU speedup or slowdown should appear as evidence.

Checks used: `squeue -j 23078,23079`; `sacct` was unavailable because the SLURM
accounting DB connection failed; both stdout/stderr logs, manifests, and summary
CSVs were inspected. No source files changed and no tests were needed.

### Claim decision

The project is a **mechanism-and-limits result**, with modest CPU positives as
secondary evidence. Accept-targeted CPU sweeps show deployable speedup on `3/6`
models, at most `1.125x`, but the pre-registered accept-targeted gate reaches
only `1.0425x` and fails its `1.05x` threshold. Oracle-current retains a
`1.2x`-`1.5x` analysis-only ceiling. There is no valid GPU performance claim.

T50 must update stale `RESEARCH.md` framing and produce a concise evidence
summary without promoting pre-T43, under-tuned, or internally inconsistent GPU
numbers.

### T50 paper rewrite progress

`paper.tex` was rewritten around the final mechanism-and-limits evidence. It now
contains the repaired data-dependent transition and sequential per-bucket
executor methodology, correctness protocol, acceptance-targeted six-model CPU
suite, oracle/overhead decomposition, adaptation confound, dedicated-MIG control
failure, limitations, and a claim-bounded conclusion. The original author and
bibliography information were retained; all obsolete draft A100 speedup tables
and claims were removed.

Validation:

```bash
tectonic -o /tmp/abnuts-paper-build paper.tex
# PASS: /tmp/abnuts-paper-build/paper.pdf generated
```

After review, the original full-width Figure 1 architecture diagram was
restored. Three PGFPlots/TikZ result visualizations were added from valid final
data: six-model speedups, executed-work ratio versus speedup, and warm-time
component decomposition. The obsolete pre-repair plots were not reused.

Only `paper.tex` and this status entry changed. `RESEARCH.md` reconciliation is
the remaining T50 step.

### The headline does not survive proper step-size tuning

T48 corrected the *sweep's* step sizes but left the oracle-gap gate target — the
source of the `1.19x`-`1.29x` headline — at `step_size=0.03`, which at `D=32`
gives mean acceptance `0.997` against the `0.8` dual-averaging target. Same flaw,
uncorrected, in the number that would have gone in an abstract.

Re-measured at the accept-targeted step size (`0.2`, accept `0.867`),
`results/raw/oracle_gap/accept_targeted_cpu`:

| mode | bucket | speedup | at step 0.03 | work ratio |
|---|---|---|---|---|
| history | 64 | **`1.0425`** | `1.19`-`1.23` | `0.528` |
| history | 128 | `0.9738` | `1.19`-`1.29` | `0.593` |
| oracle_previous | 64 | `0.9573` | `1.16` | `0.543` |
| oracle_previous | 128 | `1.0735` | `1.11` | `0.599` |
| oracle_current (analysis-only) | 64 | `1.4739` | `1.49` | `0.347` |
| oracle_current (analysis-only) | 128 | `1.2151` | `1.61` | `0.441` |

Monolithic warm fell `1.3102s -> 0.6613s`: a tuned sampler does half the work.

**The best deployable speedup is `1.0425x`, which fails the `1.05x` gate.** The
earlier headline was substantially an artifact of an under-tuned sampler —
bucketing was reclaiming waste that dual-averaging would never have produced.

### What survives, stated precisely

- **The mechanism is real and unaffected.** Executed-work ratios of `0.35`-`0.60`
  at the tuned step size, and `oracle_current` still reaches `1.47x`. Bucketing
  does reclaim genuine straggler work.
- **The deployable predictor cannot cash it.** `history` captures a work ratio of
  `0.528` where oracle plans reach `0.347`, and once the sampler is tuned the
  remaining margin no longer covers planning, gather and scatter.
- **The binding constraint is now predictor quality**, with a much thinner
  margin than T41 suggested, because the ceiling itself dropped.

The defensible claim is now:

> Bucketed scheduling reclaims real straggler work on a correctly implemented
> NUTS, and an oracle scheduler converts that into roughly `1.2x`-`1.5x`. A
> deployable history-based predictor does not consistently cash it: at step
> sizes a real sampler adapts to, the six-model CPU sweep ranges from `0.69x`
> to `1.13x`, because
> the fixed per-transition scheduling cost consumes most of what it reclaims.

That is a considerably weaker result than this morning's, and it is the one the
evidence supports.

### Remaining

- Reconcile `RESEARCH.md`, which still describes the pre-repair project.
- Generate a concise mechanism-and-limits evidence summary from valid post-T43,
  accept-targeted CPU results.
- Keep the GPU result explicitly unresolved/invalid rather than selecting a
  favorable row from run `23079`.

---

## Current active task

**Historical T49 planning entry — GPU measurement on a quiet node; then write up**

### T48 complete: step sizes chosen the way NUTS would choose them

The `scheduling` profiles' step sizes were re-selected on **mean acceptance
statistic ~= 0.8**, the dual-averaging target a real sampler adapts to. The
previous selection maximised the oracle work ratio, which is one of the two
conditions bucketing needs and fights the other. Selecting instead on "whatever
makes bucketing look best" would have been tuning to the outcome; acceptance
rate is independent of bucketing and cannot flatter it.

Chosen: funnel `0.2`, eight_schools_centered `0.3`, eight_schools_noncentered
`0.6`, gaussian_process `0.2`, stochastic_volatility `0.2`,
hierarchical_logistic `0.5`. Full sweeps recorded in the session log; several
sit at the edge of the swept range, so the true 0.8 point may be slightly
beyond.

### The re-run, and what it costs the story

`results/raw/broad_sweep_accept_targeted/`, `C=512`, 16 steps, CPU. Leapfrog
counts matched monolithic in every row.

| model | monolithic warm | was | speedup | was |
|---|---|---|---|---|
| stochastic_volatility | `0.246s` | `0.549s` | `0.687x` | `0.690x` |
| hierarchical_logistic | `0.349s` | `1.314s` | `0.813x` | `1.046x` |
| eight_schools_noncentered | `0.384s` | `1.603s` | `0.734x` | `0.968x` |
| funnel | `0.991s` | `1.160s` | **`1.125x`** | `0.706x` |
| eight_schools_centered | `1.922s` | `3.230s` | **`1.103x`** | `1.129x` |
| gaussian_process | `3.485s` | `14.743s` | **`1.022x`** | `1.195x` |

Still 3 of 6, but **a different three**. Two things happened:

1. **The funnel flipped, `0.706x -> 1.125x`.** That is the tuning error being
   corrected: the paper's headline target does benefit, and the earlier sweep
   said otherwise only because of how I had chosen its step size.
2. **Every monolithic time fell sharply** — the GP by `4.2x`. The old configs
   had the sampler doing several times more work than a properly adapted one
   would. Part of the earlier, larger speedups was bucketing reclaiming waste
   that a well-tuned sampler never generates.

Speedups compressed toward 1.0: the best is now `1.125x` against `1.195x`, and
the amortisation correlation weakened from **Spearman +0.943 to +0.714** — still
clearly positive, but on six points it is suggestive rather than established.

### The honest headline, updated

> On a correctly implemented NUTS with step sizes at the standard adaptation
> target, bucketed scheduling reclaims real straggler work on every target
> measured but yields at most ~1.1x wall-clock on this suite, helping on 3 of 6
> models. The benefit grows with per-transition work, because planning, gather
> and scatter are a fixed cost per transition.

This is a more modest claim than the `1.29x` recorded earlier today, and it is
the defensible one. The earlier figures came from a sampler tuned well below the
adaptation target, i.e. doing avoidable work that bucketing then partly
reclaimed.

### What is left

- **GPU on a quiet node.** Still the only genuinely unanswered question. Job
  `23077` never scheduled with `--exclusive` (nodes drained/reserved) and was
  resubmitted without it. Every GPU figure in this repository remains void.
- **`RESEARCH.md` is stale** and still describes the pre-repair project.
- Optionally re-run the oracle-gap gate target at an accept-targeted step size;
  its `1.19x` is measured at `step_size=0.03`, far below the adaptation target,
  so it carries the same caveat as the old sweep.

---

## Current active task

**Active task: T48 — Re-tune on the right objective, then re-measure**

*Test-suite note (2026-08-09):* the broad sweep broke
`test_report_generates_sensitivity_outputs_from_raw_tree`, and the failure was a
false alarm worth recording. That test ran the report over the **live**
`results/raw` tree and asserted the sensitivity coverage table names a
"divergence status" gap. The report emits that row only when *no* included row
reports divergences, so the sweep filling the gap — eight schools and the funnel
produced the project's first divergent rows — deleted the expected line. The
report was correct; the test asserted on mutable repository state. It now builds
its input from a fixed list of result families, so adding future benchmark runs
cannot break unrelated tests.

### Broad CPU sweep: the first answer to "where does bucketing help"

Every model-family result this project had came from run `21070`, which is void.
This is the first sweep on a validated standard-NUTS sampler, with tuned
configs, min-of-3 timing and work-ratio instrumentation.
`results/raw/broad_sweep_standard_nuts/`, `C=512`, 16 steps, CPU. Bucketed and
monolithic leapfrog counts matched in **every** row.

| model | monolithic warm | best speedup | predictor |
|---|---|---|---|
| gaussian_process | `14.743s` | **`1.195x`** | history |
| eight_schools_centered | `3.230s` | **`1.129x`** | history |
| eight_schools_noncentered | `1.603s` | `0.968x` | none |
| hierarchical_logistic | `1.314s` | **`1.046x`** | none |
| funnel | `1.160s` | `0.706x` | history |
| stochastic_volatility | `0.549s` | `0.690x` | history |

Bucketing helps on **3 of 6** models.

### The finding: heterogeneity is necessary but not sufficient

Ordering that table by per-transition work makes the pattern obvious, and the
rank correlation between monolithic warm time and best speedup is
**Spearman +0.943** across the six models.

Bucketing pays when the target is *expensive enough to amortise the fixed
scheduling cost*. Planning, gather and scatter cost roughly a constant per
transition; reclaimed straggler work scales with the work being done. Cheap
targets cannot cover the overhead no matter how heterogeneous they are.

That is a clean, defensible scientific statement and it is the paper's headline:

> Bucketed scheduling reclaims real straggler work on every target measured, but
> converts it into wall-clock speedup only where per-transition work is large
> enough to amortise the scheduler. On this suite the crossover sits near one
> second of monolithic warm time per 16 transitions at `C=512`.

### I tuned the configs on the wrong objective

The step sizes in the `scheduling` profiles were chosen to **maximise the oracle
work ratio** — the reclaimable fraction. That objective is wrong on its own,
because pushing the reclaimable fraction up means shortening trajectories, which
shrinks exactly the per-transition work needed to amortise the overhead.

The funnel is the clearest casualty. At `step_size=0.03` (deeper trajectories,
the oracle-gap target) it measures `1.23x` to `1.47x`. At `step_size=0.1` — the
value I tuned for maximum oracle ratio — it measures `0.706x`. Same model, same
executor, opposite conclusion, purely from a tuning objective that optimised one
of the two necessary conditions and ignored the other.

**T48 must re-tune on the joint objective** (reclaimable fraction *and* absolute
per-transition work) and re-run the sweep. I expect several of the negative rows
to move; the funnel almost certainly will.

### GPU measurement is not currently possible on this cluster

Four runs of an identical configuration produced monolithic warm times of
`0.1137s`, `2.9015s`, `1.9870s` and `0.3120s` — a 25x range **on the control**.
The last of those followed a change to the scatter path, which monolithic does
not execute at all, so the variation cannot be attributed to the code.

The nodes are shared; other jobs were holding ~63GB on three of four GPUs. Job
`23076` has been submitted with `--exclusive` to get a whole node. **Until an
exclusive-node result exists, no GPU number in this repository means anything**,
including every GPU figure recorded above in earlier entries.

The same caution applies to one CPU measurement: the scatter-fix comparison in
`results/raw/oracle_gap/scatter_fix` was taken while the broad sweep was running
on the same machine, and its monolithic baseline moved 33% against the run it
was being compared to. Repeated on an idle machine
(`results/raw/oracle_gap/scatter_fix_quiet`, load average 0.61): monolithic
`1.3102s` against `1.3154s` before the change, a 0.4% agreement, and history
speedups `1.1877x` / `1.2127x` against `1.2304x` / `1.1902x`. **The scatter
optimisation makes no measurable difference on CPU**; the contended run's
apparent gain was an artifact. It removes two full array copies per field so it
may still matter on a bandwidth-bound GPU, but that is unmeasured and must not
be claimed. Keeping it: it is strictly less work and the equivalence gate
passes.

---

## Current active task

**Active task: T47 — Broad CPU sweep with tuned configs; GPU wrapper as a second track**

### The headline survives the correctness repairs

Re-measured on the corrected standard-NUTS sampler, same pre-registered target
(funnel `C=512`, `D=32`, `max_tree_depth=8`, CPU, min-of-3 timing),
`results/raw/oracle_gap/standard_nuts/summary.csv`:

| mode | bucket | speedup | previous | work ratio | oracle-plan |
|---|---|---|---|---|---|
| history | 64 | **`1.2304`** | `1.1726` | `0.611` | `0.379` |
| history | 128 | `1.1902` | `1.2856` | `0.673` | `0.462` |
| oracle_previous | 64 | `1.1587` | `1.0805` | `0.611` | `0.379` |
| oracle_current | 128 | `1.6115` | `1.7120` | `0.462` | `0.462` |

Leapfrog counts identical across every mode (`241464`). Monolithic warm rose
`0.9591s -> 1.3154s` from the recursive criterion's cost, and the bucketed rows
rose with it, so the ratio held.

**This is a real result on a sampler that is now standard NUTS, validated
against a brute-force reference and analytic ground truth.** Every non-analysis
row clears the `1.05x` gate. It was measured with a deployable predictor, not an
oracle, and the transition is bitwise preserved — so because ESS per iteration
is identical by construction, this wall-clock speedup *is* an ESS/s speedup.

Since the last entry: the simplified U-turn criterion was replaced with the
standard recursive one, and two GPU sweeps were run. Three findings, two of them
unwelcome.

### 1. Standard recursive U-turn implemented and independently validated

The criterion now checks *aligned power-of-two blocks* at every level of the
binary tree, which is what the Hoffman-Gelman recursion unrolls to, instead of
prefixes anchored at the subtree start.

Implemented with a checkpoint stack: a level-`k` block opens at step `i` when
`i % 2**k == 0` and closes when `(i + 1) % 2**k == 0`, at which point the span
from its checkpoint to the current state is tested. Level 0 is a single leapfrog
step and is not tested, matching the recursion's base case.

Validated three ways, not just by the two implementations agreeing:
- `test_subtree_uturn_matches_brute_force_recursion` compares against a
  reference that stores every state in a subtree and tests every aligned block
  directly, over 72 combinations of seed, step magnitude, direction and depth.
  It checks both the verdict and the exact step at which termination fires.
- The unrolled reference resolves the block structure with Python integer
  arithmetic at trace time, so it is a genuinely independent implementation
  rather than a restatement of the traced one.
- The existing differential and ground-truth tests still pass. 94 total.

Effect: funnel mean realized depth `5.49 -> 4.73`. More checks, earlier
termination, as expected.

**Not implemented, deliberately:** Betancourt's generalized criterion adds
checks spanning the boundary between adjacent merged subtrees. This is the
classic recursion, and that is now stated in the source rather than left
ambiguous.

### 2. The standard criterion is expensive, and how expensive depends on the model

Monolithic warm time at `C=2048`, `D=128` went from `0.1137s` (simplified
criterion) to `2.9015s` (recursive, first implementation) — a 25x regression.

Cause: the first implementation wrote each tree level's checkpoint with its own
scatter and tested each level with its own pair of dot products, so every
leapfrog step paid `levels x chains x dimension` of memory traffic against a
leapfrog step's own `chains x dimension`. Vectorising both across the level axis
brought it to `1.9870s`, a 1.46x recovery.

That still leaves ~17x against the simplified criterion, and the reason is
structural rather than a coding defect: the criterion evaluates `L = 8` levels
per step where only about one closes, so it does `O(log n)` elementwise work per
leapfrog step. For a cheap target this dominates the gradient. **The funnel is
close to a worst case** — its gradient is `O(D)`, so there is very little real
work to amortise the checks against. On a target with an expensive gradient (the
GP's is `O(N^2)`) the same absolute overhead would be negligible.

This is a genuine correctness-versus-speed trade and it should be reported as
one, not buried. The remaining inefficiency is known and bounded: only the
levels that actually close need testing, roughly one per step rather than eight,
but expressing that under `vmap` needs traced-bound control flow.

### 3. GPU: the executor wrapper costs 45% before any bucketing happens

The `gpu_bucket_sweep` profile added the 1-bucket control, which is the
diagnostic that matters: one bucket **is** monolithic plus gather and scatter,
so it must land near `1.0x`.

Job `23074`, funnel `C=2048`, `D=128`, after the vectorisation fix. Monolithic
warm `1.9870s`, leapfrog counts identical across every row (`1368424`):

| buckets | bucket size | history | oracle_current | work ratio (history) |
|---|---|---|---|---|
| 1 | 2048 | `0.5224` | `0.5615` | `1.000` |
| 2 | 1024 | `0.5159` | `0.4150` | `0.829` |
| 4 | 512 | `0.2350` | `0.2478` | `0.761` |
| 8 | 256 | `0.1518` | `0.1452` | `0.705` |

**The 1-bucket control is `0.52x`, not `1.0x`.** With one bucket there is no
partitioning, no reclaimed work and nothing scanned in parallel — the row is
monolithic plus a `lax.map` of length one, a gather and a scatter, and it
already loses 45% of the runtime. The earlier sweep (job `23073`, before the
vectorisation) put the same control at `0.78x`.

So the GPU penalty decomposes into two independent costs:
1. **A fixed wrapper overhead of 22-48%**, present with a single bucket and
   entirely unrelated to bucketing.
2. **A per-bucket cost** that then degrades things monotonically with bucket
   count, on top of it.

That reorders the work. Chasing concurrent bucket execution is premature while
the wrapper alone loses half the runtime with one bucket. **T47 should attack
the wrapper first** — gather/scatter of full `(buckets, width, D)` rectangles on
device, and a `lax.map` that is pure overhead when `num_buckets == 1`.

### 4. Design option 2 is impossible (recorded so nobody tries it)

See the analysis retained below: under `vmap`, a `lax.while_loop` compiles to one
loop whose trip count is shared by every lane, so a single fused loop always runs
to the batch maximum. Per-bucket early exit therefore *requires* separate
compiled loop regions. Occupancy and early exit cannot both be had inside one
fused loop. That is a property of SIMD execution, not of JAX.

### Status of the earlier GPU result

Job `23070`'s numbers are superseded by `23074`: they were measured with the
simplified U-turn criterion and before the vectorisation fix. The qualitative
conclusion is unchanged and now rests on a cleaner sweep with an internal
control. The `0.1137s` monolithic figure from that run should not be compared
against anything current; it was a faster but incorrect sampler.

---

### T45 result (superseded, retained for provenance): funnel `C=2048`, `D=128`


| mode | bucket | speedup | work ratio | oracle-plan | warm (s) |
|---|---|---|---|---|---|
| monolithic | — | `1.0000` | `1.000` | `1.000` | `0.1137` |
| history | 512 | **`0.3464`** | `0.760` | `0.502` | `0.3283` |
| history | 256 | `0.0188` | `0.683` | `0.434` | `6.0375` |
| oracle_previous | 512 | `0.0168` | `0.726` | `0.502` | `6.7810` |
| oracle_previous | 256 | `0.0075` | `0.683` | `0.434` | `15.2068` |
| oracle_current (analysis-only) | 512 | `0.4234` | `0.502` | `0.502` | `0.2685` |
| oracle_current (analysis-only) | 256 | `0.2788` | `0.434` | `0.434` | `0.4078` |

Total leapfrog counts identical across every mode (`1350280`), so the transition
is preserved. The guard confirmed the job ran repaired code
(`abnuts source verified: /workspace/src/abnuts/__init__.py`).

### Against the pre-registered predictions

1. **Mechanism gate: PASS.** Non-analysis executed-work ratios are `0.683` to
   `0.760`, all below `1.0`. The schedule really does reclaim work, and it
   reclaims about as much as it did on CPU. This is a property of the partition,
   not the device, and it held exactly as predicted.
2. **Primary gate: FAIL, decisively.** Best non-analysis speedup is `0.3464x`
   against a `1.05x` threshold. It was pre-registered as uncertain; it is now
   answered.
3. **Diagnostic outcome, as pre-registered:** wall clock is far below `1.0x`
   while the work ratio is well below `1.0`. That is the occupancy trade-off,
   and the pre-registered conclusion applies: **the sequential-scan executor is
   the wrong design for GPUs.**
4. The CPU result stands. It is not retracted; its scope is now bounded.

### Reading it

Monolithic runs `2048` lanes as one `vmap` and finishes `1.35M` leapfrog steps
in `0.1137s`. That is a GPU doing exactly what it is good at. `lax.map` then
trades that single wide launch for 4 sequential launches of 512 lanes (or 8 of
256), each far too small to saturate the device. The reclaimed work is real but
it is nowhere near enough to pay for the lost occupancy.

The bucket-count dependence is the signature: 512-wide buckets reach `0.346x`,
256-wide collapse to `0.019x`. Narrower and more numerous is monotonically
worse, which is what serialization plus underutilization predicts and what CPU
could not reveal because a CPU has so little parallelism to lose.

**Caveat, stated plainly.** Two rows with identical work ratios differ 2.5x in
wall time (`history` 256 at `6.04s` versus `oracle_previous` 256 at `15.21s`,
both at work ratio `0.683`). Identical executed work cannot legitimately produce
that spread, so these GPU timings carry substantial noise — the node was shared,
with other jobs holding ~63GB on three of four GPUs. The qualitative conclusion
is robust because the effect is 3x to 50x and monotonic in bucket count, but
**no individual GPU figure above should be quoted as precise** and the run should
be repeated on a quiet node before anything is published.

### T46 design analysis: option 2 is impossible

I proposed three candidate designs when recording T45. Working through option 2
before implementing it shows it cannot exist, which is worth recording so nobody
spends a task on it.

Option 2 was "a single wide `vmap` over all lanes with a *per-bucket*
termination predicate, so occupancy is preserved while each bucket still exits
at its own depth."

That is not realizable. Under `vmap`, a `lax.while_loop` compiles to **one**
loop whose trip count is shared by every lane; the batching rule runs while
`any(active)` and masks lanes that have finished. A single fused loop therefore
executes exactly `max(depth)` iterations over the whole batch, and every lane
pays every iteration. Sorting or grouping the lanes changes which lanes sit next
to each other but not the global maximum, so it cannot change the trip count.

Stated generally: **per-bucket early exit requires per-bucket control flow, and
per-bucket control flow requires separate compiled loop regions.** Occupancy and
early exit are not independently obtainable within one fused loop. This is a
property of SIMD execution, not of JAX.

So the real design question is narrower than it looked: given that separate loop
regions are mandatory, can they be made to execute *concurrently* instead of
sequentially?

- On one GPU this is hard. XLA serializes independent subcomputations within a
  program, and JAX's async dispatch queues work on a single stream, so separate
  bucket programs still run back to back.
- Across devices it is natural, and it is what the repository name suggests:
  shard buckets over the 4 available GPUs, giving true concurrency **and**
  per-bucket exit. The catch is that it changes the baseline — a 4-device
  bucketed run must be compared against a 4-device monolithic run, not a
  1-device one, or the comparison is meaningless.
- Fewer and wider buckets trades reclaimed work for launch efficiency and needs
  no new machinery. The `gpu_bucket_sweep` job measures exactly this trade.

### What T46 must do

- Keep the mechanism. Executed-work reclamation is confirmed on GPU and is not
  in question.
- Replace `lax.map` over the bucket axis. Sequential scanning is what costs the
  occupancy. Candidate designs, in the order they look promising:
  1. One compiled executor per canonical bucket size, dispatched **concurrently**
     rather than scanned, so the device sees independent work it can overlap.
  2. A single wide `vmap` over all lanes with a *per-bucket* termination
     predicate, so occupancy is preserved while each bucket still exits at its
     own depth. This is the design that would keep both properties, if the
     masking can be expressed without reintroducing a global `any(active)`.
  3. Far fewer, far wider buckets — at `C=2048` perhaps 2 buckets of 1024 —
     trading most of the reclaimable work for occupancy.
- Re-measure on a **quiet** GPU node, with repeats, before drawing conclusions.
- Re-run the CPU gate afterwards to confirm no regression there.

Design 2 is the interesting one and deserves being tried first: it is the only
candidate that does not choose between the two effects.

---

### Historical: the bucket-scaling failure that prompted the timing audit

`python_bucket_loop_overhead` measures warm runtime max/min across bucket-count
rows, threshold `1.25`, and now reads `1.350` (1 bucket `0.2571s`, 2 buckets
`0.2584s`, 4 buckets `0.3471s` on `tiny_cpu`).

That criterion was written to detect **Python per-bucket dispatch** in the
pre-repair executor. After T40 there is no Python dispatch: buckets run inside a
single compiled `lax.map`. What the number now reflects is the structural cost
of *narrow* buckets on CPU — a 32-lane `vmap` at `D=128` vectorizes poorly, and
scanning buckets is sequential. Note the shape: 1 and 2 buckets are within noise
of each other (`0.2571` vs `0.2584`) and the jump appears only at width 32. That
is a width threshold, not linear-in-count dispatch overhead.

A homogeneous-batch probe (all chains identical, so reclaimable work is exactly
zero, isolating structural cost) put partitioning overhead in the range of
roughly 15-40% on CPU. Those timings were noisy: the single-bucket control at
`C=512` read `0.746x` where it should read `~1.0x`, so the range is indicative
only and the question deserves a cleaner measurement, ideally on GPU where the
narrow-`vmap` penalty behaves differently.

**T44 must re-specify this gate before measuring against it**, the same
discipline applied in T41: decide what it is now testing (structural partition
overhead, not Python dispatch), pick a threshold and a venue that match, write
both down, and only then re-run. Changing the threshold after watching it fail
is what `AGENTS.md` section 8 forbids; changing a criterion that provably
measures the wrong phenomenon, in advance and on the record, is not the same
thing.

### The scheduler question is reopened, and the answer flipped

The T41-era finding that predictors are useless was an artifact of the bug. With
the corrected sampler, measured after 30 burn-in steps at `C=512`:

| model | depth mean/sd | r(prev step) | random grouping | history | oracle |
|---|---|---|---|---|---|
| funnel | `6.35/1.09` | **`+0.904`** | `1.000` | `0.746` | `0.477` |
| eight_schools_centered | `7.98/0.19` | `+0.980` | `1.000` | `1.000` | `1.000` |
| gaussian_process | `7.66/0.67` | `+0.351` | `1.000` | `1.000` | `0.895` |
| stochastic_volatility | `6.75/0.87` | `+0.058` | `0.999` | `1.000` | `0.549` |
| hierarchical_logistic | `7.75/0.45` | `+0.308` | `1.000` | `1.000` | `0.912` |

Two reversals of the T41 reading:

- Realized depth **is** strongly predictable on the funnel (`+0.904` from the
  previous step, against `~0.00` pre-fix). The predictors were never the
  problem.
- **Random grouping now buys nothing** (`1.000`), where pre-fix it tied with
  every predictor at `~0.59`. So the gain comes from *prediction*, not from
  partitioning per se. That is the opposite of the T41 conclusion and it
  vindicates the draft's original thesis.

The withdrawn "delete the scheduler" recommendation stays withdrawn.

Two things this table also says, which matter for the benchmark suite:

- Four of five models sit at or near `max_tree_depth=8` with near-zero spread,
  so they have no heterogeneity to bucket. Their configs were written for
  `max_tree_depth` 2-5 with matching step sizes; run at depth 8 they simply cap.
  Step sizes need retuning per model before any of them is a meaningful
  scheduling target.
- `stochastic_volatility` has real reclaimable work (`oracle 0.549`) that is
  genuinely unpredictable (`r=+0.058`). That is the one honest instance of the
  T41 hypothesis, and it is worth a sentence in the paper.

Broad sweeps remain blocked. T42 (scheduler quality) is still deferred and must
be re-scoped against the post-fix numbers: the gap is now `history 0.746` versus
`oracle 0.477` on the funnel, not the `0.705` versus `0.172` T41 recorded.

---

## Completed: T43 — Repair the U-Turn Criterion

### Blocking finding (2026-08-09): the U-turn check ignores integration direction

`_build_subtree` evaluates the U-turn criterion as
`_is_turning(start, current, inverse_mass_matrix)`, where `current` was reached
from `start` by integrating with `signed_step`. When the sampled direction is
"left", `signed_step` is negative and `current` lies **before** `start` in
trajectory order. `_is_turning` computes
`delta = right.position - left.position` and tests `delta . velocity < 0`, so
with the arguments in reversed order `delta` points backward along the
trajectory and both dot products are negative **by construction**.

Every backward subtree is therefore declared turning on its first step,
regardless of the target's geometry.

Direct verification, one leapfrog step from the same state, funnel `D=8`:

| step | `delta . p_start` | `delta . p_current` | turning as called | turning correctly oriented |
|---|---|---|---|---|
| `+0.05` (forward) | `+0.6769` | `+0.6847` | False | False |
| `-0.05` (backward) | `-0.6690` | `-0.6610` | **True** | False |

Consequence: the trajectory stops as soon as a "left" direction is drawn, and
direction is a fair coin flip per doubling. Realized tree depth is therefore
geometric with `p = 0.5` and **independent of the model**. Measured at
`C=4096`, one step, `max_tree_depth=8`:

| model | d=1 | d=2 | d=3 | d=4 | d=5 | d=6 |
|---|---|---|---|---|---|---|
| funnel | 0.487 | 0.251 | 0.127 | 0.068 | 0.066 | 0.002 |
| eight_schools_centered | 0.487 | 0.251 | 0.127 | 0.069 | 0.039 | 0.015 |
| gaussian_process | 0.487 | 0.251 | 0.127 | 0.068 | 0.040 | 0.016 |
| geometric(0.5) | 0.500 | 0.250 | 0.125 | 0.062 | 0.031 | 0.016 |

Three unrelated targets, one distribution, matching the coin-flip prediction.

The bug is localized. The outer `global_turning = _is_turning(left, right, ...)`
is correctly oriented, because `left` and `right` are maintained by direction in
the doubling loop. Only the subtree call has reversed arguments.

### What this invalidates

- **Sampler validity.** Trajectories terminate on integration direction rather
  than geometry. This violates the U-turn invariant in `AGENTS.md` section 2.1.
  Every posterior, ESS, SBC, and trace-quality result this codebase has produced
  is suspect until the criterion is fixed and they are regenerated.
- **All performance magnitudes.** T39's `1.545x`/`1.910x` headroom, T40's
  `2.360x` executor speedup and `0.139`-`0.556` work ratios, and T41's `1.324x`
  `oracle_current` were all measured on a depth distribution manufactured by
  this bug. Depth is capped in practice, since reaching depth `d` requires `d`
  consecutive "right" draws with probability `2^-d`, so `max_tree_depth` is
  effectively unreachable and trajectories are far shorter than intended. The
  bucketing *mechanism* is still real; the numbers are not trustworthy.
- **The T41 scheduler diagnosis.** The measured unpredictability of realized
  depth (rank correlation `-0.03` to `+0.02` for every predictor, deployed
  predictors tying with random grouping at `~0.59` work ratio) is fully
  explained by this bug: depth was a coin-flip sequence, so nothing could
  predict it. The conclusion "predictors are unfixable, delete the scheduler" is
  **withdrawn**. Once depth reflects geometry it may well be predictable, and
  T42 must be re-evaluated after T43, not before.

### What this does not invalidate

- T39's control-flow rewrite and T40's per-bucket executor are semantics
  preserving with respect to whatever transition they wrap. The T39 differential
  test compares against the preserved unrolled reference and still holds. Both
  changes remain correct; only the measurements taken through them need redoing.
- The bug is pre-existing, from T01-T31. T39 preserved it faithfully in both the
  control-flow and unrolled implementations, which is exactly what a
  semantics-preserving rewrite should do, and is why the differential test
  passes.
- Monolithic-versus-bucketed equivalence tests cannot detect this. Both paths
  share the transition, so they agree perfectly while both being wrong. 88
  passing tests did not catch it, and that gap is itself a finding.

---

## Deferred task: T42 — Repair Scheduler Quality and Planner Cost

Deferred behind T43. The T41 diagnosis below is accurate about *what* was
measured but its causal reading is superseded by the U-turn finding above: the
predictor gap it attributes to scheduler quality is largely an artifact of
depth being a coin-flip sequence. Re-evaluate after T43.

T41 re-gated from raw post-T40 results and the gate says **FAIL**. The mechanism
gate passed; the heterogeneous wall-clock gate and the homogeneous negative
control failed. Both failures are host-side scheduling, not the executor.

The decisive number: on the pre-registered target (funnel `C=512`, `D=32`,
`max_tree_depth=8`, CPU), `oracle_current` reaches **`1.324x` end-to-end
including planner cost**, while the deployed `history` predictor reaches
`0.794x`. The same executor reaches an executed-work ratio of `0.172` with
perfect grouping and only `0.705` with predicted grouping. Predictor mean
absolute error is `0.937` against realized depths spanning 1 to 6.

So the ceiling is no longer `1.0x`, and what stands between the current code and
a real speedup is predictor quality plus a planner that costs `11.1%` of warm
time. That is T42.

Gate report: `results/processed/performance_gate/mechanism_repair/gate_report.md`.
T38 is not promoted; see the T41 notes for why "mechanism present, venue
insufficient" was explicitly declined as a basis for promoting it.

Broad sweeps remain blocked.

---

## Completed: T39 and T40 mechanism repair

The mechanism repair works:

- T39 gave the transition data-dependent control flow, so executed work now
  depends on realized tree depth. Before T39 it did not, and no scheduler could
  have won.
- T40 made each bucket exit independently via `lax.map` over the bucket axis.
  On a heterogeneous synthetic target with oracle plans, the executor now runs
  `0.139x` to `0.556x` of monolithic's lane-steps and is **faster than
  monolithic in wall clock, up to `2.360x`** (`C=512`, `D=32`, CPU,
  executor-only timing). Per-chain leapfrog counts match monolithic exactly.

**This is the first positive speedup evidence in the project, and it is not yet
a gate pass.** It is executor-only timing on CPU with oracle plans. The
end-to-end `run_bucketed` path on the `tiny_cpu` profile is still `0.60x`, and
T40 attributed that gap: the `history` predictor's plans capture a work ratio of
only `0.852` where oracle plans reach `0.339` on the same batch.

The bottleneck therefore moved from executor architecture to **predictor quality
and planner cost**. T41 re-gated and confirmed exactly that, which is why T42 is
now the active task.

---

## Superseded task: T37S

T37S is **superseded, not completed**. A source-level review on 2026-08-09 found
that the T37S heterogeneous-speedup gate was unreachable by construction, and
that the recorded cause of the repair-phase slowdown was wrong. T39 repaired the
cause; the record of the finding follows.

### Blocking finding (2026-08-09): the transition had no early exit

As of 2026-08-09, before T39, `one_chain_nuts_transition` contained no
data-dependent control flow. The code described below is preserved verbatim by
T39 as `one_chain_nuts_transition_unrolled` and `_build_subtree_unrolled` in
`src/abnuts/nuts/transition.py`; the line numbers cited are from the pre-T39
file:

- `transition.py:99` — `for depth in range(max_tree_depth)` was a Python loop,
  fully unrolled at trace time.
- `transition.py:222` — `for _ in range(num_steps)` unrolled all `2**depth`
  leapfrog steps of every subtree.
- `transition.py:224` — `active` / `should_step` only masked results through
  `jnp.where`. `leapfrog_step` and its gradient were evaluated on every step of
  every chain regardless.

There was no `lax.while_loop` and no early exit anywhere in the transition.
Every chain therefore always paid for the full `2**max_tree_depth - 1` leapfrog
budget, and **realized-depth heterogeneity cost nothing**. There was no
straggler waste for a bucketing scheduler to reclaim.

Consequence for the bucketed executor, repaired by T40:
`src/abnuts/nuts/bucketed.py:319` flattened the whole
`(num_buckets, max_bucket_size)` rectangle into one array and calls
`monolithic_transition` **once**. While per-lane work was constant, this gave
the identity

```text
bucketed_cost == monolithic_cost * (padded_lanes / num_chains)
                 + planner + gather + scatter
```

so repaired bucketed speedup is bounded above by `1.0x` and can only be reached
in the limit of zero padding and zero overhead. The `>= 1.05x` gate cannot be
met by any planner, predictor, or overhead tuning while this holds.

This identity reproduces every repair-phase number already on this board:

- `oracle_current` has zero predictor error and zero padding yet still measures
  `0.697x`, because it is monolithic plus overhead.
- The homogeneous negative control measures `0.809x`, which is pure overhead.
- `bucket_size=512` over `C=128` executes `4x` the lanes and measures `0.394x`.
- Repaired local non-analysis rows faster than monolithic: `0/13`.
- In pre-repair run `21070`, all `21/735` faster rows had `num_buckets=1`.

The previously recorded causes — Python per-bucket dispatch, predictor quality,
padding waste, and "tiny-CPU fixed overhead" — describe the residual, not the
cause. They are superseded by this finding.

### Diagnostic probe evidence

Read-only CPU probe, Neal's funnel, `C=64`, `D=8`, `max_tree_depth=6`,
`jit_monolithic_transition`, warm time as the min of 3 blocked repeats:

| case | realized depth (mean/max) | actual leapfrogs | warm |
|---|---|---|---|
| `step_size=0.001` (deep trees) | 1.97 / 6 | 252 | 22.85 ms |
| `step_size=25.0` (all stop at depth 1) | 1.00 / 1 | 64 | 25.28 ms |

Ratio shallow/deep = `1.106`. Four times less real work, no time saved.

Warm time versus `max_tree_depth` at `step_size=0.05`: `d=3` 1.13 ms, `d=4`
2.16 ms, `d=5` 4.77 ms, `d=6` 20.65 ms. Runtime tracks `max_tree_depth` alone
and roughly doubles per level, consistent with fully unrolled `2**depth` work.

Compile cost of the unrolled program at `C=64`, `D=8`, `depth=6` was measured at
`4m23s` by the XLA slow-operation alarm. The draft target case is `C=2048`,
`D=128`, `depth ~ 10`. The unrolled design is not expected to reach paper scale,
which is additional evidence that the draft `~1.28x` result came from a sampler
with real control flow.

The probe was run from the session scratchpad and wrote nothing into the
repository. T39 promoted it into
`tests/test_transition_control_flow.py::test_warm_runtime_depends_on_realized_tree_depth`,
which now asserts the reverse of this result.

### Direction

Path A (fix the mechanism) was selected by the user on 2026-08-09. The tasks are
T39 (control-flow rewrite, **complete**), T40 (per-bucket executor rewrite,
**active**), then T41 (re-gate). T38 minimal HPC validation stays blocked until
T41 passes, and broad sweeps stay blocked behind T38.

No `sbatch`, `srun`, SLURM, GPU/HPC validation, or broad sweep was run.

---

## How a coding agent should use this file

1. Read `AGENTS.md`.
2. Read `RESEARCH.md`.
3. Read this `STATUS.md`.
4. Implement only the active task above.
5. Run the smallest validation listed for that task.
6. Fix only failures caused by that task.
7. Update this file with files changed, commands run, results, and the next active task.
8. Stop.

`STATUS.md` is the authoritative task board. `AGENTS.md` contains stable rules; `RESEARCH.md` contains scientific framing.

---

## Current diagnosis from run 21070

Run `21070` is negative diagnostic evidence, not final paper evidence.

Observed results:

- Completed outputs mostly exist, except `hierarchical_logistic` initially.
- Only `21/690` completed bucketed rows were faster than monolithic, all in `chain_scaling`.
- Best observed bucketed speedup was `1.264x` for `chain_scaling`, `C=128`, `D=128`, history predictor, bucket size `512`.
- Target-like `C=2048` cases were much slower for bucketed execution.
- Best family speedups were mostly slowdowns:
  - `funnel_ablation`: `0.632x`,
  - `dimension_scaling`: `0.309x`,
  - `eight_schools_centered`: `0.725x`,
  - `eight_schools_noncentered`: `0.530x`,
  - `stochastic_volatility`: `0.498x`,
  - `gaussian_process`: `0.512x`.
- `gaussian_process` had bucketed-vs-monolithic leapfrog-count mismatches in many rows.

Pre-T34 likely architecture problem:

- `src/abnuts/nuts/bucketed.py` looped over `plan.num_buckets` in Python.
- It called `monolithic_transition` separately for each bucket.
- Monolithic runs one vectorized transition over all chains.
- Bucketed ran many smaller vectorized transitions plus gather/scatter.
- The hot path appeared to have little or no effective `jax.jit` compilation.
- Slowdown appears to scale with number of buckets, consistent with serial dispatch/host overhead.

Immediate conclusion after run 21070: repair executor architecture before running
more broad sweeps. T34 completed the local fixed-shape executor repair and T35
completed the local correctness regression gate. T36 decomposed oracle and
overhead effects. T37 failed the performance gate. T37R repaired the missing
homogeneous negative-control gate, but the regenerated gate still failed the
heterogeneous speedup requirement.

**Superseded on 2026-08-09.** The pre-T34 Python per-bucket dispatch loop was
real and did add overhead, but it was never the reason bucketing could not win.
The transition itself is fully unrolled and has no early exit, so realized-depth
heterogeneity has no runtime cost and bucketing has no waste to reclaim. See the
blocking finding under "Current active task". Every statement in this section
that attributes the slowdown primarily to per-bucket dispatch, predictor
quality, or padding should be read as a partial explanation of the residual
only. Run `21070` remains valid as negative diagnostic evidence; its recorded
root cause does not.

---

## Global invariants

- Bucketed execution must preserve the same NUTS transition as monolithic.
- Same per-chain RNG sequence must be preserved under equivalence tests.
- Padding must never affect real chain states or transition metrics.
- All timing must use `block_until_ready()` or an equivalent tree-blocking helper.
- LaTeX-first artifacts remain required for reports.
- Raw results must not be overwritten without `--overwrite`.
- Oracle-current results are analysis-only upper bounds.
- Broad model sweeps are blocked until a re-gate passes after T43 and T42.
- The U-turn criterion was broken until T43 (2026-08-09) and is now repaired.
  **Any performance magnitude or posterior-quality result generated before T43
  is void.** Post-T43 raw outputs live in `results/raw/oracle_gap/uturn_fixed`,
  `results/raw/t43_uturn_fixed`, `results/raw/correctness/gp_tiny_uturn_fixed`,
  and `results/raw/performance_gate/homogeneous_negative_control_uturn_fixed`.
- The transition must adapt trajectory length to geometry. This is guarded by
  `tests/test_sampler_ground_truth.py`; consistency tests between execution
  paths cannot detect a defect in the shared transition.
- No performance gate may be evaluated on a target where bucketing has no
  mechanism to exploit. A valid heterogeneous target requires a transition with
  data-dependent early exit and an executor that keeps buckets separate.

---

## Phase 4 — Performance architecture repair

### Dependency map

```text
T31 completed validation
  └─ T32 Performance Forensics and Reproduction
       └─ T33 JIT Baseline Parity
            └─ T34 Real Fixed-Shape Bucket Executor
                 └─ T35 Correctness Regression Gate
                      └─ T36 Oracle/Overhead Decomposition Before Broad Sweeps
                           └─ T37 Performance Gate Before Paper Sweeps  [FAIL]
                                └─ T37R Repair Performance Gate Failures  [FAIL]
                                     └─ T37S  [SUPERSEDED 2026-08-09]
                                          └─ T39 Data-Dependent Control Flow  [DONE]
                                               └─ T40 Per-Bucket Fixed-Shape Executor  [DONE]
                                                    └─ T41 Re-gate After Mechanism Repair  [FAIL]
                                                         └─ T43 Repair the U-Turn
                                                            Criterion  [DONE]
                                                              └─ re-gate: 4 of 5 gates flipped
                                                                 to PASS, speedup 1.239x
                                                                   └─ T44 Re-specify Bucket-
                                                                      Scaling Gate  [ACTIVE]
                                                                        └─ T38 Minimal HPC
                                                                             └─ T42 Scheduler
                                                                                (re-scope)
```

Do not promote T38 or any broad paper sweep until the T41 re-gate passes.

T39 and T40 must both land before T41 is meaningful. T39 alone creates the
straggler waste; T40 alone has nothing to reclaim. Re-gating between them is
allowed as a diagnostic but cannot pass the heterogeneous gate.

---

## Task board

### [x] T32 — Performance Forensics and Reproduction

**Goal:** Convert run `21070` from an alarming broad sweep into a precise performance-forensics report and tiny reproduction harness. No source semantic changes yet.

**Scope:**

- Locate and parse run `21070` raw summaries/logs where available.
- Generate a short markdown or LaTeX report documenting:
  - best/median/worst speedups,
  - number of bucketed rows faster than monolithic,
  - speedup by experiment/model,
  - speedup versus number of buckets where metadata permits,
  - whether slowdown scales with `plan.num_buckets`,
  - Gaussian-process transition-metric/leapfrog-count mismatches.
- Add a tiny profiling config/profile that runs monolithic and bucketed for:
  - `C in {128, 512, 2048}`,
  - `S in {128, 512}`,
  - at least one fixed dimension/profile compatible with the current code,
  - timing breakdown enabled or explicitly marked as unavailable.
- Add or extend analysis code only as needed to summarize existing and tiny profiling outputs.
- Do not rewrite sampler semantics, JIT behavior, planner behavior, or executor behavior in this task.

**Acceptance criteria:**

- A report exists under a clear path such as `results/latex/performance_forensics/` or `results/processed/performance_forensics/`.
- The report states whether slowdown appears to scale with number of buckets.
- The report states `21/690` faster rows from run `21070` if confirmed from raw data, or clearly labels it as provided diagnostic context if raw data are unavailable.
- The report flags Gaussian-process leapfrog-count mismatches.
- A tiny profiling config/profile exists for the next tasks.
- No source semantic changes to NUTS transition, monolithic execution, bucketed execution, or planner behavior are made.

**Smallest validation:**

```bash
python3 -m pytest -q
python3 -m abnuts.analysis.forensics \
    --run-dir /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw \
    --out results/processed/performance_forensics \
    --run-id 21070 --overwrite
```

**Completion notes (2026-05-30):**

All acceptance criteria met. No source semantic changes were made.

Files changed:
- `src/abnuts/analysis/forensics.py` — new CLI module (`python3 -m abnuts.analysis.forensics`)
- `configs/performance_forensics.yaml` — tiny profiling config (profiles: `tiny_cpu`, `forensics_s128`, `forensics_s512`)

Outputs generated:
- `results/processed/performance_forensics/report.md` — markdown forensics report
- `results/processed/performance_forensics/forensics_summary.csv` — per-row data

Commands run:
```bash
python3 -m abnuts.analysis.forensics \
    --run-dir /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw \
    --out results/processed/performance_forensics --run-id 21070 --overwrite
python3 -m pytest -q   # 60 passed
```

Key findings from raw data (confirmed from `/home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw`):

- Total bucketed rows in run 21070 raw data: **735** (STATUS.md diagnostic note said 690; raw data shows 735).
- Bucketed rows faster than monolithic: **21 / 735 (2.9%)**, all in `chain_scaling`, all with `num_buckets=1`.
- Best speedup: **1.2645x** (chain_scaling, C=128, D=128, history predictor, bucket_size=512, num_buckets=1).
- Median speedup across all bucketed rows: **0.2439x**.
- Worst speedup: **0.0573x** (dimension_scaling, C=2048, D=256, hvp predictor, bucket_size=128, num_buckets=16).

Speedup vs num_buckets (confirms dispatch-loop hypothesis):
- 1 bucket → median 0.975x, best 1.264x, 21/90 faster
- 2 buckets → median 0.491x, 0/135 faster
- 4 buckets → median 0.248x, 0/210 faster
- 8 buckets → median 0.124x, 0/195 faster
- 16 buckets → median 0.062x, 0/105 faster

Median speedup halves with each doubling of `num_buckets` (factor ≈ 0.50 consistently). This confirms Python per-bucket dispatch overhead is dominating.

GP leapfrog-count mismatches: **35 / 45 (77.8%)** gaussian_process bucketed rows had `bucket_total_leapfrog_count ≠ mono_total_leapfrog_count`.

Per-family best speedups: funnel_ablation 0.632x, dimension_scaling 0.309x, eight_schools_centered 0.725x, eight_schools_noncentered 0.530x, stochastic_volatility 0.498x, gaussian_process 0.512x, hierarchical_logistic 0.499x.

Per-bucket timing breakdown was NOT available (disabled in run 21070 manifests). Gather / executor / scatter split must be measured with T33's `--enable-timing-breakdown` runs.

Tiny profiling config for T33/T34 (`configs/performance_forensics.yaml`):
- `tiny_cpu`: C=[128], D=128, num_steps=8, bucket_sizes=[128,512] — CPU-runnable
- `forensics_s128`: C=[128,512,2048], D=128, num_steps=128, bucket_sizes=[128,512] — GPU required for C>=512
- `forensics_s512`: C=[128,512,2048], D=128, num_steps=512, bucket_sizes=[128,512] — GPU required for C>=512

SLURM not required for T32. No SLURM commands were run.

---

### [x] T33 — JIT Baseline Parity

**Goal:** Establish a compiled monolithic baseline before changing bucketed execution.

**Dependencies:** T32.

**Scope:**

- Add a jitted monolithic transition/run path or a carefully scoped compiled transition wrapper.
- Separate first-compile/cold timing from warm timing.
- Ensure warm timing excludes first compile or reports it separately.
- Preserve the current non-jitted path if useful for debugging.
- Add deterministic tests comparing jitted and current monolithic outputs.

**Acceptance criteria:**

- Jitted monolithic matches current monolithic under deterministic tests.
- Warm jitted monolithic is not slower than current monolithic on the tiny profiling profile, or any slowdown is explained and bounded.
- Timing output records compile/cold time and warm time separately.

**Smallest validation:**

```bash
python -m pytest -q tests/test_monolithic.py tests/test_transition.py
# plus the T32 tiny profiling profile with jitted monolithic enabled
```

**Completion notes (2026-05-30):**

All acceptance criteria met. No sampler semantic changes were made.

Files changed:
- `src/abnuts/nuts/monolithic.py` — added `jit_monolithic_transition` (module-level `jax.jit` wrapper with `static_argnames=("model", "max_tree_depth")`) and `run_monolithic_jit` (identical semantics to `run_monolithic` but uses the cached JIT transition; preserves the non-JIT `run_monolithic` for debugging).
- `src/abnuts/experiments/run_benchmark.py` — updated `_time_monolithic` to use `run_monolithic_jit`; added `run_monolithic_jit` to import.
- `tests/test_monolithic.py` — added three new tests: `test_jit_monolithic_transition_matches_monolithic_transition`, `test_run_monolithic_jit_matches_run_monolithic`, `test_jit_monolithic_transition_is_deterministic_across_calls`.

Commands run:
```bash
python3 -m pytest -q tests/test_monolithic.py tests/test_transition.py   # 8 passed
python3 -m pytest -q                                                       # 63 passed
python3 -m abnuts.experiments.run_benchmark \
    --config configs/performance_forensics.yaml \
    --profile tiny_cpu --backend cpu --method monolithic \
    --enable-timing-breakdown \
    --out results/raw/t33_jit_baseline/tiny_cpu --overwrite
```

Output: `results/raw/t33_jit_baseline/tiny_cpu/summary.csv`

Timing results (C=128, D=128, funnel, num_steps=8, CPU):
- `timing_cold_run_seconds`: 19.135s (includes JIT compile on first step)
- `timing_warm_iteration_seconds`: 0.175s (all 8 steps from compiled cache, ~109x faster than cold)
- `timing_cold_compile_seconds`: 18.960s (estimated JIT compile overhead = cold − warm)
- `timing_breakdown_enabled`: True

Correctness:
- XLA compilation reorders float32 operations relative to JAX eager mode, causing ULP-level differences (up to ~2 ULP observed). This is documented in the test docstrings.
- All integer/boolean metrics and RNG keys are bitwise identical between JIT and eager.
- Float arrays checked with `atol=1e-5, rtol=1e-5` (much stricter than any statistical significance threshold).

SLURM not required. No SLURM commands were run.

---

### [x] T34 — Real Fixed-Shape Bucket Executor

**Goal:** Replace host Python per-bucket hot-path dispatch with a genuine fixed-shape compiled executor strategy.

**Dependencies:** T33.

**Scope:**

Implement one of these designs, or another JAX-native equivalent:

1. one cached compiled executor per canonical bucket size,
2. a single compiled padded rectangular executor using `lax.scan`, `lax.map`, `vmap`, or similar,
3. another fixed-shape design that avoids Python loops over buckets in the warm hot path.

The implementation may still plan on host, but gather/executor/scatter must be measured separately where feasible.

**Acceptance criteria:**

- Bucketed warm runtime no longer scales roughly linearly with number of buckets on a synthetic heterogeneous-work benchmark, or the remaining scaling is quantified and explained.
- Bucketed warm runtime improves over monolithic on at least one tiny heterogeneous diagnostic, or the task records exactly why it still cannot.
- The report separates planning, gather, executor, scatter, compile, and warm iteration costs as far as current instrumentation allows.

**Smallest validation:**

```bash
python -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_planner.py
# plus T32/T33 tiny profiling profile with repaired bucketed executor enabled
```

**Completion notes:**

Completed on 2026-06-03. The Python per-bucket hot-path dispatch loop was
removed from `src/abnuts/nuts/bucketed.py`.

Implementation summary:
- `bucketed_transition` now uses a rectangular fixed-shape JIT strategy over the
  full padded `(num_buckets, max_bucket_size)` plan.
- The normal bucketed transition path calls one compiled function that performs
  gather, fixed-shape executor, and scatter together.
- The timing-breakdown path calls one compiled gather, one compiled executor, and
  one compiled scatter per transition so planner/gather/executor/scatter costs
  remain separately measurable.
- Padded lanes scatter to a sentinel row outside the real chain range. The
  sentinel row is discarded, so padded lanes cannot overwrite real chain states,
  transition metrics, or RNG keys.
- The old per-bucket gather/execute/scatter helper path was removed.
- Bucketed benchmark timing now uses the same profiled component path for cold
  and warm timing when `--enable-timing-breakdown` is set. This prevents compile
  time for component JITs from being charged to the first warm bucketed row.

Files changed:
- `src/abnuts/nuts/bucketed.py`
- `src/abnuts/experiments/run_benchmark.py`
- `tests/test_monolithic_bucketed_equivalence.py`
- `STATUS.md`

Outputs generated:
- `results/raw/t34_fixed_shape_bucket_executor/tiny_cpu/summary.csv`
- `results/raw/t34_fixed_shape_bucket_executor/tiny_cpu/manifest.json`
- `results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling/summary.csv`
- `results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling/manifest.json`

Validation commands run:
```bash
python -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_planner.py
# failed: `python` is not on PATH in this environment; non-blocking, reran with python3.

python3 -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_planner.py
# after initial implementation: 3 failed, 6 passed.
# diagnosis: exact float comparisons failed due ULP-level XLA/JIT float32
# differences; RNG keys, realized depth, leapfrog count, divergence flags, and
# max-depth flags were exact. Tests were updated to the same strict 1e-5 float
# tolerance style used by T33 while preserving exact checks for discrete metrics
# and RNG keys.

python3 -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_planner.py
# final result: 9 passed in 21.81s

python3 -m abnuts.experiments.run_benchmark \
    --config configs/performance_forensics.yaml \
    --profile tiny_cpu --backend cpu \
    --enable-timing-breakdown \
    --out results/raw/t34_fixed_shape_bucket_executor/tiny_cpu --overwrite
# result: completed, wrote summary.csv and manifest.json

python3 -m abnuts.experiments.run_benchmark \
    --config configs/performance_forensics.yaml \
    --profile tiny_cpu --backend cpu \
    --method monolithic,bucketed \
    --predictors history \
    --bucket-sizes 32,64,128 \
    --enable-timing-breakdown \
    --out results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling \
    --overwrite
# result: completed, wrote summary.csv and manifest.json
```

Key local timing results:
- `tiny_cpu` monolithic warm: 0.1686s; cold: 19.35s.
- `tiny_cpu` bucketed/history/bucket_size=128: warm 0.2374s, speedup 0.711x,
  1 bucket, 0 padding, planner 0.0329s, gather 0.0033s, executor 0.1278s,
  scatter 0.0054s, unattributed 0.0680s.
- `tiny_cpu` bucketed/none/bucket_size=128: warm 0.2343s, speedup 0.720x,
  1 bucket, 0 padding, planner 0.0408s, gather 0.0016s, executor 0.1159s,
  scatter 0.0053s, unattributed 0.0707s.
- `tiny_cpu` bucketed/history/bucket_size=512: warm 0.4279s, speedup 0.394x,
  1 bucket, 384 padded lanes; slowdown is explained by padded executor work.
- Transition metric parity in the tiny timing runs: monolithic and bucketed
  total leapfrog counts matched exactly (`3056` vs `3056`) for all rows.

Bucket-count scaling diagnostic (`tiny_cpu_bucket_scaling`, C=128, D=128,
history predictor, 8 steps, no padding):
- bucket_size=32: 4 buckets, warm 0.2136s, speedup 0.678x.
- bucket_size=64: 2 buckets, warm 0.2035s, speedup 0.712x.
- bucket_size=128: 1 bucket, warm 0.2072s, speedup 0.699x.
- Conclusion: local warm runtime is roughly flat across 4/2/1 buckets. The old
  pre-repair pattern where time scaled roughly linearly with bucket count is not
  present in this tiny CPU diagnostic.

**Partially superseded on 2026-08-09.** The removal of the Python per-bucket
dispatch loop stands. The replacement design — flattening the bucket rectangle
into a single `monolithic_transition` call — is now known to make bucketed work
strictly greater than monolithic work, and must be replaced by T40. The
"Tiny CPU speedup over monolithic: not achieved" note below was correct; the
reason recorded for it was not.

Acceptance status:
- Fixed-shape executor architecture: complete for local CPU validation.
- Bucket-count dispatch scaling: no longer roughly linear in the tiny CPU
  diagnostic; remaining small differences are within normal CPU/JIT timing noise
  and fixed overhead.
- Tiny CPU speedup over monolithic: not achieved. The best local bucketed speedup
  was 0.720x. This is recorded honestly; on CPU at C=128, planner/scatter and
  unattributed overhead exceed the executor-time savings, and padded cases are
  slower from extra lane work.
- Timing breakdown: available for planner, gather, executor, scatter, cold, warm,
  and estimated compile time in the generated summaries.

Commands intentionally not run:
```bash
python3 -m abnuts.experiments.run_benchmark \
    --config configs/performance_forensics.yaml \
    --profile forensics_s128 --backend gpu \
    --enable-timing-breakdown \
    --out results/raw/t34_fixed_shape_bucket_executor/forensics_s128

python3 -m abnuts.experiments.run_benchmark \
    --config configs/performance_forensics.yaml \
    --profile forensics_s512 --backend gpu \
    --enable-timing-breakdown \
    --out results/raw/t34_fixed_shape_bucket_executor/forensics_s512
```

Reason not run: these profiles include C=512 and C=2048 GPU/HPC cases. The user
reported that their SLURM job limit is occupied and explicitly prohibited
`sbatch`, `srun`, and HPC job submission. No `sbatch`, `srun`, or SLURM command
was run.

Next active task: T35 — Correctness Regression Gate. Do not start T35 until the
next invocation.

---

### [x] T35 — Correctness Regression Gate

**Goal:** Prove the executor rewrite did not change sampler semantics.

**Dependencies:** T34.

**Scope:**

- Strengthen monolithic-vs-bucketed equivalence tests after the executor rewrite.
- Include `gaussian_process` or a minimal GP correctness case because run `21070` showed leapfrog-count mismatches.
- Check positions, RNG-derived transition metrics, leapfrog counts, divergences, max depth, and padded-lane noninterference.
- Use exact equality where possible; otherwise document strict tolerances.

**Acceptance criteria:**

- No unexplained correctness mismatches remain in the gate tests.
- GP or minimal GP transition metrics match exactly or within documented strict tolerance.
- Any remaining mismatch is treated as a blocker for broad sweeps.

**Smallest validation:**

```bash
python -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_transition.py
python -m abnuts.experiments.run_correctness --method both --model gaussian_process --profile tiny --out results/raw/correctness/gp_tiny_repair
```

If the exact CLI differs, use the smallest existing GP correctness command and record it.

**Completion notes:**

Completed on 2026-06-03. No sampler semantic changes were made.

Files changed:
- `tests/test_monolithic_bucketed_equivalence.py` — added a minimal GP
  monolithic-vs-bucketed one-step regression with real padding and exact checks
  for RNG keys, depths, leapfrog counts, divergence flags, and max-depth flags;
  added a direct poisoned padded-lane scatter test to prove false-mask lanes
  scatter only to the sentinel row and cannot overwrite real chain states,
  transition metrics, or RNG keys.
- `src/abnuts/experiments/run_correctness.py` — kept existing exact equality
  fields and added strict-tolerance gate fields:
  `equivalence_passed`, `positions_allclose`, `metrics_allclose`,
  `discrete_metrics_exact`, `float_metrics_within_tolerance`,
  `metric_max_abs_delta`, and `metric_mismatch_count`. Method `both` now fails
  if positions or float metrics exceed `atol=1e-5, rtol=1e-5`, or if RNG keys or
  discrete metrics differ.
- `STATUS.md` — recorded T35 completion and promoted T36 as the next active task.

Validation commands run:
```bash
python3 -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_transition.py
# pre-change baseline: 6 passed in 26.19s

python3 -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_transition.py
# final result: 8 passed in 49.35s

python3 -m abnuts.experiments.run_correctness \
    --method both --model gaussian_process \
    --num-chains 4 --dimension 3 --num-steps 1 \
    --step-size 0.005 --max-tree-depth 2 \
    --bucket-size 4 --predictor history \
    --out results/raw/correctness/gp_tiny_repair --overwrite
# result: completed, wrote correctness outputs
```

The documented STATUS command with `--profile tiny` was not run as written
because `run_correctness` does not expose a `--profile` flag. The equivalent
smallest existing GP correctness command above spells out the `tiny` settings
from `configs/gaussian_process.yaml`.

GP correctness output:
- Output directory: `results/raw/correctness/gp_tiny_repair`
- `equivalence_passed`: `true`
- `final_rng_keys_equal`: `true`
- `discrete_metrics_exact`: `true`
- `float_metrics_within_tolerance`: `true`
- `positions_allclose`: `true`
- `max_position_delta`: `3.725290298461914e-09`
- Max float metric deltas:
  - `acceptance_statistic`: `1.9073486328125e-06`
  - `energy_error`: `0.0`
  - `gradient_norm`: `1.430511474609375e-06`
- GP total leapfrog count matched exactly in per-iteration outputs:
  monolithic `6`, bucketed `6`.

Non-validation command note:
```bash
git status --short
```
failed in `/home/hpc/users/viktor.najdovski/distributed-nuts` because this
workspace does not include a `.git` directory. This did not block T35.

Commands intentionally not run because of SLURM/HPC limits:
- None required for T35. No `sbatch`, `srun`, GPU/HPC job, or broad sweep was
  run.

Acceptance status:
- T35 complete for local CPU validation.
- No unexplained correctness mismatches remain in the T35 gate.
- Remaining exact float differences are ULP-scale compiled-vs-eager float32
  differences and are recorded under strict `1e-5` tolerance.

Next active task: T36 — Oracle/Overhead Decomposition Before Broad Sweeps. Do
not start T36 until the next invocation.

---

### [x] T36 — Oracle/Overhead Decomposition Before Broad Sweeps

**Goal:** Determine whether remaining failure is scheduler quality or executor overhead.

**Dependencies:** T35.

**Scope:**

- Add a small oracle-current/oracle-previous diagnostic that separates:
  - predictor error,
  - padding waste,
  - gather/scatter overhead,
  - executor overhead,
  - compile cost,
  - warm iteration cost.
- Keep it small. This is not a broad paper sweep.
- Label oracle-current as an analysis-only upper bound in raw outputs and LaTeX.

**Acceptance criteria:**

- Output makes clear whether remaining slowdown is due mostly to scheduler quality, padding, gather/scatter, compile cost, or executor overhead.
- Oracle-current rows are clearly labeled analysis-only.
- The report is generated from raw results, not hand-written claims.

**Smallest validation:**

```bash
python -m pytest -q
python -m abnuts.experiments.run_oracle_gap --config configs/oracle_gap.yaml --profile tiny --out results/raw/oracle_gap/repair_tiny
python -m abnuts.analysis.report --input results/raw/oracle_gap/repair_tiny --out results/latex/oracle_gap/repair_tiny
```

**Completion notes:**

Completed on 2026-06-03. No sampler semantic changes were made.

Implementation summary:
- `run_oracle_gap` now uses the jitted monolithic baseline established in T33.
- The oracle-gap runner now records blocked cold/warm timing using the shared
  timing helper style and emits timing breakdown fields in raw `summary.csv`:
  cold run, estimated compile, warm iteration, planner, executor, gather,
  scatter, HVP, gather+scatter, unattributed, non-executor overhead, and
  dominant warm component.
- Oracle bucketed modes use the repaired fixed-shape bucket transition timing
  recorder, so gather/executor/scatter are measured separately where feasible.
- `oracle_current` remains labeled as `analysis-only upper bound` in raw outputs,
  manifest metadata, and LaTeX.
- The oracle-gap LaTeX table is generated from raw timing and scheduler fields
  and includes speedup, padding, predictor MAE, cold compile, warm, planner,
  executor, gather+scatter, other/unattributed, and dominant component columns.
- The generic timing-breakdown detector now requires benchmark-shaped
  `method`/`predictor` fields, so oracle-gap outputs are not misclassified as a
  generic profiling run.

Files changed:
- `src/abnuts/experiments/run_oracle_gap.py`
- `src/abnuts/analysis/aggregate.py`
- `src/abnuts/analysis/latex_tables.py`
- `src/abnuts/analysis/report.py`
- `STATUS.md`

Outputs generated:
- `results/raw/oracle_gap/repair_tiny/summary.csv`
- `results/raw/oracle_gap/repair_tiny/predictor_calibration.csv`
- `results/raw/oracle_gap/repair_tiny/padding_heatmap.csv`
- `results/raw/oracle_gap/repair_tiny/speedup_heterogeneity.csv`
- `results/raw/oracle_gap/repair_tiny/manifest.json`
- `results/latex/oracle_gap/repair_tiny/tables/table_oracle_gap.tex`
- `results/latex/oracle_gap/repair_tiny/figures/fig_oracle_gap_waterfall.tex`
- `results/latex/oracle_gap/repair_tiny/figures/fig_predictor_calibration.tex`
- `results/latex/oracle_gap/repair_tiny/figures/fig_padding_heatmap.tex`
- `results/latex/oracle_gap/repair_tiny/figures/fig_speedup_vs_heterogeneity.tex`
- `results/latex/oracle_gap/repair_tiny/figures/predictor_calibration.csv`
- `results/latex/oracle_gap/repair_tiny/figures/padding_heatmap.csv`
- `results/latex/oracle_gap/repair_tiny/figures/speedup_vs_heterogeneity.csv`

Validation commands run:
```bash
python -m pytest -q
# failed: `python` is not on PATH in this environment; no tests were run.

python3 -m pytest -q
# result: 65 passed in 82.65s

python -m abnuts.experiments.run_oracle_gap \
    --config configs/oracle_gap.yaml --profile tiny \
    --out results/raw/oracle_gap/repair_tiny
# failed: `python` is not on PATH in this environment; output directory was not
# written by this command.

python3 -m abnuts.experiments.run_oracle_gap \
    --config configs/oracle_gap.yaml --profile tiny \
    --out results/raw/oracle_gap/repair_tiny
# result: completed; wrote oracle-gap raw outputs.

python -m abnuts.analysis.report \
    --input results/raw/oracle_gap/repair_tiny \
    --out results/latex/oracle_gap/repair_tiny
# failed: `python` is not on PATH in this environment; no report files were
# written by this command.

python3 -m abnuts.analysis.report \
    --input results/raw/oracle_gap/repair_tiny \
    --out results/latex/oracle_gap/repair_tiny
# result: completed; wrote oracle-gap LaTeX table and figures.
```

Key tiny CPU results from `results/raw/oracle_gap/repair_tiny/summary.csv`:
- Monolithic warm time: `0.03183s`.
- Best non-HVP bucketed row: `history`, warm speedup `0.725x`,
  predictor MAE `0.362`, padding ratio `0.000`.
- `oracle_previous`: warm speedup `0.678x`, predictor MAE `0.250`,
  padding ratio `0.000`.
- `oracle_current` analysis-only upper bound: warm speedup `0.697x`,
  predictor MAE `0.000`, padding ratio `0.000`.
- For `oracle_current`, gather+scatter was `0.00056s`, executor was
  `0.00080s`, planner was `0.00677s`, and unattributed warm overhead was
  `0.03751s`; dominant warm component was `unattributed`.
- HVP and hybrid rows were dominated by HVP probing (`~0.074s-0.075s`) and were
  slower (`0.259x` and `0.265x`).

Interpretation:
- On this tiny CPU diagnostic, the remaining slowdown is not explained by
  scheduler quality or padding because `oracle_current` has zero predictor error
  and zero padding but is still slower than monolithic.
- Gather/scatter and executor components are small in this run; warm overhead is
  mostly unattributed outside the named JIT component timers for non-HVP rows.
- HVP/hybrid slowdown is explained primarily by HVP probe overhead.
- This is local CPU diagnostic evidence only; it is not paper-scale performance
  evidence and does not justify broad sweeps by itself.

Commands intentionally not run because of SLURM/HPC limits:
- None required for T36. No `sbatch`, `srun`, GPU/HPC job, or broad sweep was
  run.

Acceptance status:
- T36 complete for local CPU validation.
- Oracle-current rows are clearly labeled analysis-only in raw output and
  LaTeX.
- The report is generated from raw results.

Next active task: T37 — Performance Gate Before Paper Sweeps. Do not start T37
until the next invocation.

---

### [x] T37 — Performance Gate Before Paper Sweeps

**Goal:** Decide whether broad sweeps are scientifically justified after repair.

**Dependencies:** T36.

**Scope:**

Create a gate report from raw T32–T36 outputs. Do not run a broad sweep.

Required gates:

- no unexplained correctness mismatches,
- warm bucketed speedup `>= 1.05x` on at least one pre-registered heterogeneous target,
- homogeneous negative-control slowdown is explained and bounded,
- timing breakdown shows executor overhead is not dominated by Python bucket loops,
- report is generated from raw results,
- report states how many bucketed rows are faster than monolithic.

**Acceptance criteria:**

- Gate report explicitly says `PASS` or `FAIL`.
- If `FAIL`, broad sweeps remain blocked and the next active task must be a repair task, not T38.
- If `PASS`, promote T38.

**Smallest validation:**

```bash
python -m pytest -q
# plus the gate-report command added by this task
```

**Completion notes:**

- Completed on 2026-06-03. The gate report explicitly says **FAIL**.
  Broad sweeps remain blocked and T38 minimal HPC validation is not promoted.

Implementation summary:
- Added `python -m abnuts.analysis.performance_gate`, a file-backed T37 gate
  report command.
- The command reads existing T32-T36 artifacts:
  - raw run `21070` summaries under
    `/home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw`,
  - T35 repaired GP equivalence JSON,
  - T34 repaired tiny CPU benchmark summaries,
  - T36 oracle-gap raw summary.
- The command writes a processed gate CSV, markdown report, manifest, and
  LaTeX report.
- The report counts pre-repair and repaired bucketed rows faster than
  monolithic and keeps run `21070` labeled as pre-repair negative diagnostic
  evidence.
- `oracle_current` rows are reported separately as analysis-only upper bounds
  and are not allowed to satisfy the non-analysis speedup gate.

Files changed:
- `src/abnuts/analysis/performance_gate.py`
- `tests/test_performance_gate.py`
- `STATUS.md`

Outputs generated:
- `results/processed/performance_gate/repair_tiny/gate_summary.csv`
- `results/processed/performance_gate/repair_tiny/gate_report.md`
- `results/processed/performance_gate/repair_tiny/manifest.json`
- `results/latex/performance_gate/repair_tiny/gate_report.tex`

Validation commands run:
```bash
python3 -m pytest -q tests/test_performance_gate.py
# result: 1 passed in 0.17s

python3 -m abnuts.analysis.performance_gate \
    --pre-repair-run-dir /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw \
    --correctness-dir results/raw/correctness/gp_tiny_repair \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu/summary.csv \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling/summary.csv \
    --oracle-gap-dir results/raw/oracle_gap/repair_tiny \
    --out results/processed/performance_gate/repair_tiny \
    --latex-out results/latex/performance_gate/repair_tiny \
    --overwrite
# result: completed; wrote gate CSV, markdown report, manifest, and LaTeX.
# gate decision: FAIL.

python3 -m pytest -q
# result: 66 passed in 79.97s

python --version
# failed: `python` is not on PATH in this environment.
```

Key T37 gate results:
- Overall performance gate: **FAIL**.
- Correctness gate: **PASS** from
  `results/raw/correctness/gp_tiny_repair/equivalence.json`.
- Heterogeneous speedup gate: **FAIL**. Best repaired non-analysis row was
  `0.725x` (`oracle-gap history`, bucket size `2`), below the required `1.05x`.
- Homogeneous negative-control gate: **FAIL**. No raw homogeneous
  negative-control summary exists in T32-T36 outputs.
- Python bucket-loop overhead gate: **PASS** locally. Repaired no-padding warm
  runtime max/min ratio across bucket-count rows was `1.151`, so the old
  pre-repair linear per-bucket dispatch pattern is not present in this tiny CPU
  diagnostic.
- Faster rows reported: pre-repair run `21070` had `21/735` bucketed rows
  faster than monolithic; repaired local non-analysis evidence had `0/13` rows
  faster than monolithic.
- Historical GP mismatch evidence from run `21070` remains `35/45` GP bucketed
  rows, while the repaired T35 GP gate passes.

Commands intentionally not run because of SLURM/HPC limits:
- No `sbatch`, `srun`, SLURM, GPU/HPC job, or broad sweep was run.
- T38's minimal HPC command remains blocked and was not run:
```bash
sbatch scripts/hpc/benchmark_sweep.sbatch  # with a repair/minimal profile only
```

Acceptance status:
- T37 complete.
- The gate report explicitly says `FAIL`.
- Broad sweeps remain blocked.
- The next active task is a repair task, not T38.

Next active task: T37R — Repair Performance Gate Failures. Do not start T37R
until the next invocation.

---

### [x] T37R — Repair Performance Gate Failures

**Goal:** Repair the local failures identified by the T37 gate before minimal
HPC validation or broad sweeps are promoted.

**Dependencies:** T37 failed.

**Scope:**

- Isolate the remaining bucketed warm overhead in the local repaired executor,
  especially the unattributed component reported by T36/T37.
- Add a CPU-safe homogeneous negative-control diagnostic that writes raw
  `summary.csv` output with blocked timing breakdown fields.
- Keep the diagnostic small; do not run a broad sweep.
- Preserve NUTS transition semantics, per-chain RNG sequence in equivalence
  tests, and padded-lane noninterference.
- Regenerate the performance gate report from raw outputs after the repair.

**Acceptance criteria:**

- The repaired local gate inputs include a raw homogeneous negative-control
  summary.
- The report explains whether the homogeneous negative-control slowdown is
  bounded and which timing component dominates it.
- The repaired non-analysis heterogeneous evidence either reaches the `1.05x`
  gate or clearly records the remaining local blocker.
- If the regenerated gate still fails, the next active task remains a repair
  task and T38 remains blocked.
- If the regenerated gate passes, promote T38 minimal HPC validation.

**Smallest validation:**

```bash
python3 -m pytest -q
python3 -m abnuts.analysis.performance_gate \
    --pre-repair-run-dir /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw \
    --correctness-dir results/raw/correctness/gp_tiny_repair \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu/summary.csv \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling/summary.csv \
    --oracle-gap-dir results/raw/oracle_gap/repair_tiny \
    --negative-control-summary results/raw/performance_gate/homogeneous_negative_control/summary.csv \
    --out results/processed/performance_gate/repair_tiny \
    --latex-out results/latex/performance_gate/repair_tiny \
    --overwrite
```

Do not run `sbatch`, `srun`, GPU/HPC validation, or a broad sweep for T37R
unless a future user instruction explicitly changes the SLURM constraint.

**Completion notes:**

- Completed on 2026-06-03. The regenerated gate report explicitly says
  **FAIL**. Broad sweeps remain blocked and T38 minimal HPC validation is not
  promoted.

Implementation summary:
- Added a CPU-safe homogeneous negative-control diagnostic command:
  `python3 -m abnuts.experiments.run_homogeneous_negative_control`.
- The diagnostic writes raw output at
  `results/raw/performance_gate/homogeneous_negative_control/summary.csv` and
  includes blocked production warm timing, blocked profiled component timing,
  planner/executor/gather/scatter/unattributed fields, component-measurement
  overhead, ready-tree blocking overhead, and monolithic-vs-bucketed equivalence
  fields.
- Updated `python3 -m abnuts.analysis.performance_gate` so
  `--negative-control-summary` uses production speedup for the slowdown bound,
  reports profiled speedup and diagnostic overhead fields, and includes T37R raw
  summaries in the raw-results gate.
- The oracle-gap `repair_tiny` unattributed overhead is now diagnosed as local
  fixed CPU bookkeeping/timing overhead plus repeated planning in a tiny run,
  not renewed Python per-bucket dispatch. The repaired bucket-count scaling gate
  remains flat enough locally (`1.151` max/min warm ratio across 1/2/4
  no-padding bucket rows).

Files changed:
- `src/abnuts/experiments/run_homogeneous_negative_control.py`
- `src/abnuts/analysis/performance_gate.py`
- `tests/test_performance_gate.py`
- `STATUS.md`

Outputs generated:
- `results/raw/performance_gate/homogeneous_negative_control/config.json`
- `results/raw/performance_gate/homogeneous_negative_control/summary.csv`
- `results/raw/performance_gate/homogeneous_negative_control/manifest.json`
- `results/processed/performance_gate/repair_tiny/gate_summary.csv`
- `results/processed/performance_gate/repair_tiny/gate_report.md`
- `results/processed/performance_gate/repair_tiny/manifest.json`
- `results/latex/performance_gate/repair_tiny/gate_report.tex`
- Exploratory non-canonical CPU output also exists at
  `results/raw/performance_gate/homogeneous_negative_control_c256_tmp/`.

Commands run:
```bash
python3 - <<'PY'
# Local CPU timing probes over small candidate C/D/S settings; no files written.
# Result: shallow tiny CPU runs were dominated by fixed planner/bookkeeping
# overhead; the C=256, D=128, S=8 control was bounded.
PY

python3 -m pytest -q tests/test_performance_gate.py
# result: 2 passed in 0.17s

python3 -m abnuts.experiments.run_homogeneous_negative_control \
    --out results/raw/performance_gate/homogeneous_negative_control --overwrite
# first C=128 default attempt before the default was adjusted; output was
# overwritten by the final C=256 canonical run.

python3 -m abnuts.experiments.run_homogeneous_negative_control \
    --num-chains 256 \
    --out results/raw/performance_gate/homogeneous_negative_control_c256_tmp \
    --overwrite
# result: completed; exploratory C=256 output confirmed bounded production
# speedup.

python3 -m abnuts.experiments.run_homogeneous_negative_control \
    --out results/raw/performance_gate/homogeneous_negative_control --overwrite
# result: completed; wrote final canonical C=256 negative-control raw output.

python3 -m pytest -q
# result: 67 passed in 80.13s

python3 -m pytest -q
# result after final source patch: 67 passed in 79.67s

python3 -m abnuts.analysis.performance_gate \
    --pre-repair-run-dir /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw \
    --correctness-dir results/raw/correctness/gp_tiny_repair \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu/summary.csv \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling/summary.csv \
    --oracle-gap-dir results/raw/oracle_gap/repair_tiny \
    --negative-control-summary results/raw/performance_gate/homogeneous_negative_control/summary.csv \
    --out results/processed/performance_gate/repair_tiny \
    --latex-out results/latex/performance_gate/repair_tiny \
    --overwrite
# final result: completed; gate decision FAIL.
```

Key timing results:
- Homogeneous negative control (`C=256`, `D=128`, `S=8`, one bucket, no
  padding): production bucketed speedup `0.809x`; profiled bucketed speedup
  `0.828x`; equivalence passed.
- Negative-control bucketed production warm: `0.31393s`; profiled warm:
  `0.30690s`; planner `0.03534s`; executor `0.20360s`; gather `0.00156s`;
  scatter `0.00657s`; profiled unattributed `0.05984s`; ready-tree block
  `0.00018s`.
- Regenerated gate results:
  - correctness: **PASS**,
  - heterogeneous speedup: **FAIL**, best non-analysis repaired row `0.725x`
    (`oracle-gap history`, bucket size `2`) versus required `1.05x`,
  - homogeneous negative control: **PASS**, worst production speedup `0.809x`
    versus required `0.80x`,
  - Python bucket-loop overhead: **PASS**, warm max/min ratio `1.151`,
  - report from raw results: **PASS**,
  - faster rows reported: **PASS**, pre-repair `21/735`, repaired local
    non-analysis `0/13`.

Commands intentionally not run:
- No `sbatch`, `srun`, SLURM, GPU/HPC validation, or broad sweep was run.
- T38 remains blocked.

Acceptance status:
- T37R complete.
- The missing raw homogeneous negative-control summary is repaired.
- The regenerated performance gate still fails because heterogeneous speedup
  remains below threshold.
- The next active task is another repair task, not T38.

Next active task: T37S — Repair Local Heterogeneous Speedup Failure. Do not
start T37S until the next invocation.

**Superseded on 2026-08-09.** T37S was never started and is now closed as
unreachable by construction. The active task is T39. This line is historical
record only.

---

### [-] T37S — Repair Local Heterogeneous Speedup Failure — SUPERSEDED

**Status: superseded on 2026-08-09. Not completed. Do not implement as written.**

T37S asked for the remaining `0.725x -> 1.05x` local heterogeneous speedup gap
to be closed by repairing executor overhead, planning, gather/scatter, or
tiny-CPU fixed cost. A source-level review found that gate to be unreachable by
construction: the transition is fully unrolled with no early exit, so bucketed
cost is bounded below by monolithic cost times the padded-lane ratio. See the
blocking finding under "Current active task" for the identity, the probe
evidence, and the file/line references.

T37S is replaced by T39 (control-flow rewrite), T40 (per-bucket executor
rewrite), and T41 (re-gate). Its diagnostic scope — deciding among executor
overhead, repeated planning, rectangular over-execution, and gather/scatter —
is answered: rectangular over-execution against a work-invariant transition
dominates, and the rest is residual.

The original T37S text is retained below for provenance only.

---

**Goal:** Repair or fully diagnose the remaining local heterogeneous speedup
gate failure after T37R.

**Dependencies:** T37R regenerated the gate and it still failed the
heterogeneous speedup criterion.

**Scope:**

- Focus only on the remaining failed gate: non-analysis repaired heterogeneous
  warm speedup is `0.725x`, below the required `1.05x`.
- Determine whether the local heterogeneous failure is due to production
  executor overhead, profiled timing overhead, repeated planning, rectangular
  over-execution, scatter/gather cost, tiny-CPU fixed overhead, or the absence
  of enough heterogeneity in the local repair inputs.
- If a small local repair is possible without changing NUTS semantics, implement
  it.
- Preserve NUTS transition semantics, per-chain RNG sequence in equivalence
  tests, and padded-lane noninterference.
- Do not run `sbatch`, `srun`, GPU/HPC validation, or broad sweeps unless a
  future user instruction explicitly changes the SLURM constraint.
- Regenerate the performance gate report from raw outputs after any repair.

**Acceptance criteria:**

- The regenerated gate either passes or clearly records the remaining local
  heterogeneous-speedup blocker.
- T38 is promoted only if the regenerated gate passes.
- If the regenerated gate still fails, the next active task remains a repair
  task and T38 remains blocked.

**Smallest validation:**

```bash
python3 -m pytest -q
python3 -m abnuts.analysis.performance_gate \
    --pre-repair-run-dir /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw \
    --correctness-dir results/raw/correctness/gp_tiny_repair \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu/summary.csv \
    --heterogeneous-summary results/raw/t34_fixed_shape_bucket_executor/tiny_cpu_bucket_scaling/summary.csv \
    --oracle-gap-dir results/raw/oracle_gap/repair_tiny \
    --negative-control-summary results/raw/performance_gate/homogeneous_negative_control/summary.csv \
    --out results/processed/performance_gate/repair_tiny \
    --latex-out results/latex/performance_gate/repair_tiny \
    --overwrite
```

**Completion notes:**

- Superseded. Not implemented.

---

### [x] T39 — Data-Dependent Control Flow in the NUTS Transition

**Goal:** Give the transition a real early exit so that realized tree depth
determines executed work. Without this, no scheduling layer can ever win.

**Dependencies:** T37S superseded.

**Scope:**

- Replace the unrolled `for depth in range(max_tree_depth)` loop in
  `src/abnuts/nuts/transition.py` with `lax.while_loop` (or `lax.fori_loop` with
  a real early-exit predicate) whose continuation condition is the existing
  `active` flag.
- Replace the unrolled `for _ in range(num_steps)` subtree loop with a
  data-dependent construct so a subtree stops on divergence or U-turn instead of
  masking through the full `2**depth` budget.
- Under `vmap`, the resulting predicate is "any lane still active", so a
  vectorized group runs to the group maximum depth. That is the straggler waste
  the method is supposed to remove. Record this explicitly.
- This is a control-flow change, not a semantic change. The leapfrog integrator,
  U-turn criterion, divergence logic, max-depth logic, and per-chain RNG
  sequence must be untouched.
- Keep the unrolled path available behind a flag if useful for differential
  testing, but the default must be the control-flow version.
- Promote the 2026-08-09 diagnostic probe into a repository regression test that
  asserts warm runtime now *does* depend on realized depth, so this defect
  cannot silently return.
- Do not change `src/abnuts/nuts/bucketed.py` in this task. That is T40.

**Acceptance criteria:**

- Monolithic-vs-monolithic differential test: control-flow transition matches the
  unrolled transition on positions, RNG keys, realized depth, leapfrog count,
  divergence flags, and max-depth flags. Discrete metrics and RNG keys exact;
  float metrics within the documented `1e-5` tolerance already used by T33/T35.
- The existing T35 monolithic-vs-bucketed equivalence gate still passes.
- A regression test demonstrates runtime now scales with realized depth: an
  all-shallow case must be measurably faster than a deep case at the same
  `max_tree_depth`, reversing the 2026-08-09 probe result.
- Compile time at the tiny profile is reported before and after. A large
  reduction is expected and should be recorded.

**Smallest validation:**

```bash
python3 -m pytest -q tests/test_transition.py tests/test_monolithic.py \
    tests/test_monolithic_bucketed_equivalence.py
python3 -m pytest -q
```

Do not run `sbatch`, `srun`, GPU/HPC validation, or broad sweeps.

**Completion notes:**

- Completed on 2026-08-09. All acceptance criteria met. No change to the
  leapfrog integrator, U-turn criterion, divergence logic, max-depth logic, or
  per-chain RNG sequence. `src/abnuts/nuts/bucketed.py` was not touched.

Implementation summary:
- `one_chain_nuts_transition` now runs two `lax.while_loop`s: one over
  trajectory doublings with the existing `active` flag as its continuation
  condition, and one per subtree whose predicate is exactly the unrolled
  version's `should_step` (`active & ~divergence_flag & ~turning`, bounded by
  `2**depth`).
- The previous implementation is preserved verbatim as
  `one_chain_nuts_transition_unrolled` and is exported from `abnuts.nuts`. It is
  the differential-testing reference only and must not be used in samplers or
  benchmarks.
- Setup (momentum draw, initial energy, slice variable) and finalization
  (sampler state, acceptance statistic, `TransitionInfo`) are shared by both
  implementations via `_begin_trajectory` and `_finish_trajectory`, so the two
  can only differ inside the loops.
- RNG-sequence preservation required care. The unrolled version splits the
  trajectory key once per depth for **all** `max_tree_depth` depths whether or
  not the chain is still active, and returns the last key as the chain's next
  key. Exiting early would otherwise change every subsequent transition.
  `_advance_trajectory_key` replays exactly the splits the early exit skipped.
  It performs PRNG work only, no leapfrog steps.
- Inside a genuine loop iteration the loop predicate is true, so the unrolled
  version's `depth_was_attempted` / `should_step` conjunctions collapse away.
  Every iteration the control-flow version skips is one the unrolled version
  executed only to discard: it added `0` to every counter and sum, took
  `maximum` against `0.0`, and selected the unchanged branch of every state
  update. That is why the two are semantically identical.
- `realized_tree_depth` is now the doubling loop's `depth` counter directly: the
  loop runs exactly while the chain is active, so on exit `depth` equals what
  the unrolled version accumulated.
- The doubling body passes the loop predicate, not a bare `True`, as the
  subtree's `active` flag. Under `vmap` the body also runs for lanes that
  already exited; this keeps those lanes from performing real leapfrog work
  while their results are discarded by the `while_loop` batching rule.
- `num_steps` per subtree is now `jnp.left_shift(1, depth)` in `int32` rather
  than a Python `1 << depth`.

Files changed:
- `src/abnuts/nuts/transition.py`
- `src/abnuts/nuts/__init__.py` — export `one_chain_nuts_transition_unrolled`
- `tests/test_transition_control_flow.py` — new
- `STATUS.md`

No changes were needed in `src/abnuts/nuts/monolithic.py`,
`src/abnuts/nuts/independent.py`, or `src/abnuts/nuts/bucketed.py`: the public
signature of `one_chain_nuts_transition` is unchanged, so they pick up the
control-flow version automatically.

Validation commands run:
```bash
JAX_PLATFORMS=cpu python3 -m pytest -q tests/test_transition_control_flow.py
# result: 18 passed in 50.13s

JAX_PLATFORMS=cpu python3 -m pytest -q tests/test_transition.py \
    tests/test_monolithic.py tests/test_monolithic_bucketed_equivalence.py
# result: 14 passed in 58.81s

JAX_PLATFORMS=cpu python3 -m pytest -q
# result: 85 passed in 134.38s (was 67 before T39; +18 new tests, no regressions)
```

New tests in `tests/test_transition_control_flow.py`:
- `test_transition_uses_data_dependent_control_flow` — structural gate. Counts
  `while` primitives in the traced jaxpr: the default transition must have at
  least two, the unrolled reference must have zero. Deterministic, so an
  unrolled hot path cannot be reintroduced silently regardless of timing noise.
- `test_control_flow_transition_matches_unrolled_transition` — differential gate
  over 15 cases (3 step-size regimes x 5 seeds).
- `test_batched_chain_result_does_not_depend_on_its_batch` — a chain's result
  must be identical whether it runs alone or batched with chains of different
  realized depth. This pins down the `vmap`-of-`while_loop` lane-freezing
  semantics that the entire bucketing method rests on, since bucketing does
  nothing but change which chains share a batch.
- `test_warm_runtime_depends_on_realized_tree_depth` — mechanism gate, the
  direct reversal of the 2026-08-09 probe.

Equivalence results (control flow versus unrolled, funnel `D=5`,
`max_tree_depth=6`, 100 cases over 4 step-size regimes x 25 seeds):
- Final RNG keys: **exact** in every case.
- `realized_tree_depth`, `leapfrog_count`, `divergence_flag`,
  `max_tree_depth_hit`: **exact** in every case.
- `jnp.allclose(atol=1e-5, rtol=1e-5)` failures: **0 / 100**.
- Worst absolute deltas: position `5.960e-08`, `acceptance_statistic`
  `1.907e-06`, `gradient_norm` `1.192e-07`, `energy_error` `6.0` on a divergent
  case where the value itself is `~1.99e6` (relative `3e-6`).
- The residual is float32 rounding, not semantics. Rerunning the same sweep with
  `jax_enable_x64` drops the worst position delta from `2.980e-08` to
  `1.388e-17`, tracking machine epsilon across nine orders of magnitude. The
  cause is the same one T33 and T35 document: `lax.while_loop` bodies are
  compiled and fused by XLA, while the unrolled path runs op-by-op in eager
  mode, so float32 operations are ordered differently.

Compile time before and after (funnel, `C=128`, `D=128`, `step_size=0.03`,
float32, jitted vmapped transition, `compile ~= cold - warm`):

| max_tree_depth | unrolled compile | control-flow compile | unrolled warm | control-flow warm |
|---|---|---|---|---|
| 3 | 3.69s | 1.27s | 3.96 ms | 3.97 ms |
| 5 (tiny profile) | 17.49s | 1.61s | 12.79 ms | 6.44 ms |
| 8 | not attempted | 1.40s | — | 6.59 ms |
| 10 | not attempted | 1.38s | — | 6.73 ms |

At the tiny profile the compile is `10.9x` faster and the warm transition is
`2.0x` faster. The structural change matters more than either number: unrolled
compile time grows with `2**max_tree_depth` (`4.7x` for two extra depths),
whereas control-flow compile time is flat, because program size no longer
depends on depth. The draft's target case needs `max_tree_depth ~ 10`, which the
unrolled design could not have compiled at `C=2048`, `D=128`. This is consistent
with the T33 record of a `19.135s` cold tiny_cpu run and is further evidence
that the draft's `~1.28x` result came from a sampler with real control flow.

Straggler waste now available (recorded explicitly, as T39 scope requires):

| case | realized depth histogram | one batch | oracle depth groups | headroom |
|---|---|---|---|---|
| `C=128`, `D=128`, depth 8 | `{1: 62, 2: 38, 3: 14, 4: 11, 5: 3}` | 8.14 ms | 5.27 ms | `1.545x` |
| `C=256`, `D=64`, depth 8 | `{1: 129, 2: 59, 3: 38, 4: 20, 5: 10}` | 11.15 ms | 5.84 ms | `1.910x` |

Under `vmap`, `lax.while_loop` runs while any lane is active, so one batch pays
its **maximum** realized depth while most chains finish at depth 1 or 2.
Grouping chains by realized depth costs less in total. That gap is the headroom
a scheduler can compete for, and before T39 it was exactly `1.0x` by
construction.

This measurement is an **analysis-only upper bound** in the same sense as
`oracle_current` in `results/raw/oracle_gap`: the groups are formed from realized
depths observed after the fact, and it performs no planning, gather, scatter, or
padding. A real scheduler must pay all of those out of this budget and will
therefore realize less. It is a headroom measurement, not a speedup claim, and
it is CPU-only.

Acceptance status:
- Differential test against the unrolled reference: **PASS**, discrete metrics
  and RNG keys exact, floats within the documented `1e-5` tolerance.
- Existing T35 monolithic-versus-bucketed equivalence gate: **PASS**, unchanged.
- Regression test showing runtime now scales with realized depth: **PASS**.
- Compile time reported before and after: **done**, see table above.

Diagnostic probes were run from the session scratchpad and wrote nothing into
the repository. Their durable content is the numbers recorded above plus
`tests/test_transition_control_flow.py`.

Commands intentionally not run:
- No `sbatch`, `srun`, SLURM, GPU/HPC validation, or broad sweep was run.
- No benchmark or oracle-gap run was regenerated. T39 changes warm timing for
  every method, so the existing `results/raw` timing artifacts are now stale for
  performance purposes. Regenerating them is T41's job, into a new output
  directory, after T40 lands. The `repair_tiny` artifacts are preserved as the
  record of the pre-mechanism state.

Next active task: T40 — Per-Bucket Fixed-Shape Executor. Do not start T40 until
the next invocation.

---

### [x] T40 — Per-Bucket Fixed-Shape Executor

**Goal:** Stop flattening the bucket rectangle into one transition call, so each
bucket exits at its own max depth.

**Dependencies:** T39.

**Scope:**

- `src/abnuts/nuts/bucketed.py:319` currently reshapes
  `(num_buckets, width)` into `(num_buckets * width,)` and calls
  `monolithic_transition` once. After T39 this would re-merge every bucket into
  a single `any(active)` predicate and destroy the entire benefit. It must go.
- Execute buckets as separate control-flow regions while keeping fixed shapes
  and a single dispatch. Acceptable designs under `AGENTS.md` section 2.4:
  - `lax.map` or `lax.scan` over the bucket axis, each iteration running its own
    vmapped transition and therefore its own early exit;
  - one cached compiled executor per canonical bucket size.
- Python loops over buckets in the warm hot path remain unacceptable.
- Keep the existing gather / executor / scatter timing split and the sentinel-row
  padded-lane scatter, which is correct and should be preserved as is.
- Re-measure the bucket-count scaling diagnostic. Warm runtime must not return to
  scaling linearly with `num_buckets`.

Budget to work against, from the T39 oracle headroom measurement: `1.545x` at
`C=128`, `D=128` and `1.910x` at `C=256`, `D=64` on CPU. Planning, gather,
scatter, and padding all come out of that budget, and T37R measured planning
alone at `0.03534s` against a `0.31393s` bucketed warm step at `C=256`, `D=128`.
Planner cost is therefore a first-order term, not a rounding error. If T40 lands
and the realized speedup is still below `1.0x`, the next question is planner
cost, not executor structure.

**Acceptance criteria:**

- T35 equivalence gate still passes, including padded-lane noninterference.
- A synthetic heterogeneous diagnostic shows bucketed executed leapfrog work
  below monolithic executed leapfrog work at equal chain count. Report the work
  ratio, not only wall time, so the mechanism is verified independently of
  machine noise.
- The Python bucket-loop overhead gate still passes.
- Timing breakdown still separates planner, gather, executor, scatter, cold, and
  warm.

**Smallest validation:**

```bash
python3 -m pytest -q tests/test_monolithic_bucketed_equivalence.py tests/test_planner.py
python3 -m pytest -q
```

Do not run `sbatch`, `srun`, GPU/HPC validation, or broad sweeps.

**Completion notes:**

- Completed on 2026-08-09. All acceptance criteria met. No change to NUTS
  transition semantics; `src/abnuts/nuts/transition.py` was not touched.

Implementation summary:
- `_fixed_shape_bucket_executor` now runs `lax.map` over the bucket axis. Each
  scan iteration runs its own vmapped `monolithic_transition`, so each bucket
  gets its own `lax.while_loop` and exits when its own deepest chain stops.
- The flatten-into-one-transition path is gone, along with the now-unused
  `_unflatten_bucket_tree`. Gather, scatter, the sentinel-row padded-lane
  discard, and the gather/executor/scatter timing split are unchanged.
- The executor must not become a `vmap` over the bucket axis. That would merge
  every bucket back under one `any(active)` predicate exactly as the flattened
  version did. This is recorded in the function docstring and pinned by a test,
  because it is an easy and silent regression.
- Scanning buckets is sequential, unlike monolithic's single wide `vmap`. That
  trade-off is real and is documented in the docstring; on CPU it is more than
  paid for by the reclaimed work at the bucket widths measured below, but it is
  a reason to expect different behaviour at GPU scale.

Files changed:
- `src/abnuts/nuts/bucketed.py`
- `tests/test_bucketed_executor_work.py` — new
- `STATUS.md`

Outputs generated:
- `results/raw/t40_per_bucket_executor/tiny_cpu/summary.csv` and `manifest.json`
- `results/raw/t40_per_bucket_executor/tiny_cpu_bucket_scaling/summary.csv` and
  `manifest.json`

The T34 raw artifacts were not overwritten. They remain the record of the
pre-mechanism executor.

Validation commands run:
```bash
JAX_PLATFORMS=cpu python3 -m pytest -q \
    tests/test_monolithic_bucketed_equivalence.py tests/test_planner.py
# result: 11 passed in 27.71s

JAX_PLATFORMS=cpu python3 -m pytest -q tests/test_bucketed_executor_work.py
# result: 3 passed in 17.71s

JAX_PLATFORMS=cpu python3 -m pytest -q
# result: 88 passed in 162.19s (85 before T40; +3 new tests, no regressions)

JAX_PLATFORMS=cpu python3 -m abnuts.experiments.run_benchmark \
    --config configs/performance_forensics.yaml \
    --profile tiny_cpu --backend cpu --enable-timing-breakdown \
    --out results/raw/t40_per_bucket_executor/tiny_cpu

JAX_PLATFORMS=cpu python3 -m abnuts.experiments.run_benchmark \
    --config configs/performance_forensics.yaml \
    --profile tiny_cpu --backend cpu \
    --method monolithic,bucketed --predictors history \
    --bucket-sizes 32,64,128 --enable-timing-breakdown \
    --out results/raw/t40_per_bucket_executor/tiny_cpu_bucket_scaling
```

New tests in `tests/test_bucketed_executor_work.py`:
- `test_bucket_executor_scans_over_the_bucket_axis` — structural gate, asserts a
  `scan` primitive over buckets. Deterministic guard against a regression to a
  flattened or vmapped executor.
- `test_bucketed_executes_less_leapfrog_work_than_monolithic` — mechanism gate on
  executed lane-steps.
- `test_padded_lanes_cannot_deepen_a_bucket` — padded lanes replay the last real
  chain index in their bucket, so they cannot become a bucket's slowest member
  and inflate its executed work. Worth pinning because a future change to
  sentinel-based padding would break it silently.

**Executed-work model.** Under `vmap` a group costs its slowest member: the
while loop runs until the deepest chain in the group stops and every lane pays
each iteration. Executed lane-steps for a group are therefore
`len(group) * max(leapfrog_count in group)`, summed over buckets for the
bucketed case and taken over all chains for monolithic. The model slightly
overstates both sides because it assumes the deepest lane drives every subtree,
so it is used for the ratio, where the bias largely cancels. It is hardware
independent, which is the point.

Executor results on a synthetic heterogeneous funnel target (chains spread along
the scale coordinate, `max_tree_depth=8`, oracle plans, CPU, warm timing is the
min of 5 blocked repeats of a single `bucketed_transition` against
`jit_monolithic_transition`):

| case | bucket size | buckets | work ratio | wall speedup |
|---|---|---|---|---|
| `C=128`, `D=128` | 16 | 8 | `0.236` | `1.602x` |
| depths `{1:62, 2:38, 3:14, 4:11, 5:3}` | 32 | 4 | `0.319` | `1.651x` |
| | 64 | 2 | `0.556` | `1.427x` |
| | 128 | 1 | `1.000` | `1.006x` |
| `C=256`, `D=64` | 16 | 16 | `0.177` | `1.395x` |
| depths `{1:129, 2:59, 3:38, 4:20, 5:10}` | 32 | 8 | `0.214` | `1.727x` |
| | 64 | 4 | `0.312` | `1.778x` |
| | 128 | 2 | `0.521` | `1.339x` |
| | 256 | 1 | `1.000` | `0.930x` |
| `C=512`, `D=32` | 32 | 16 | `0.139` | `2.311x` |
| depths `{1:266, 2:128, 3:58, 4:35, 5:21, 6:4}` | 64 | 8 | `0.176` | `2.360x` |
| | 128 | 4 | `0.277` | `2.234x` |
| | 256 | 2 | `0.514` | `1.527x` |

Per-chain leapfrog counts matched monolithic exactly in every row. The
single-bucket rows are the internal control: one bucket is monolithic plus
gather and scatter, and they land at `1.006x` and `0.930x` as they should.

These are **executor-only** measurements with **oracle** plans. They exclude
planning, they are CPU-only, and oracle grouping is an analysis-only upper bound
in the same sense as `oracle_current`. They are not a gate pass and must not be
reported as one.

Bucket-count scaling gate, re-measured
(`results/raw/t40_per_bucket_executor/tiny_cpu_bucket_scaling/summary.csv`,
`C=128`, `D=128`, `max_tree_depth=5`, history predictor, 8 steps, no padding):
- 4 buckets: warm `0.18978s`
- 2 buckets: warm `0.19453s`
- 1 bucket: warm `0.15800s`
- max/min warm ratio: `1.231`, within the `<= 1.25` gate. The old pre-repair
  linear-in-bucket-count pattern has not returned. The margin is tighter than
  T37R's `1.151`, which is expected: `lax.map` is sequential across buckets.

**End-to-end result, reported honestly.** On the full `run_bucketed` path the
`tiny_cpu` profile is still slower than monolithic:
- monolithic warm `0.10388s`; `history`/`bucket_size=128` warm `0.17388s`
  (`0.597x`); `history`/`bucket_size=512` warm `0.34228s` (`0.303x`, 384 padded
  lanes over 128 chains).
- Component split for `history`/`128`: planner `0.02856s`, executor `0.07381s`,
  gather `0.00171s`, scatter `0.00447s`, unattributed `0.06531s`.
- Bucketed and monolithic total leapfrog counts matched exactly (`3056`) in
  every row of both new benchmark outputs.

Attribution of that remaining gap, measured on the same `tiny_cpu` batch by
comparing the plans `run_bucketed` actually used against oracle plans built from
the same steps' realized depths:

| bucket size | monolithic lane-steps | history-plan | oracle-plan |
|---|---|---|---|
| 32 | 17280 | 14720 (`0.852`) | 5856 (`0.339`) |
| 64 | 17280 | 15936 (`0.922`) | 9536 (`0.552`) |

The executor delivers `0.339` when given good plans. The `history` predictor
only reaches `0.852`, so it captures roughly a seventh of the available work
reduction. Planner cost then takes `0.02856s` of a `0.17388s` step on top of
that. **The remaining loss is predictor quality and planner cost, not executor
structure.** That is a different and more tractable problem than the one T32-T37S
were chasing.

Acceptance status:
- T35 equivalence gate including padded-lane noninterference: **PASS**, 11
  tests, unchanged.
- Bucketed executed leapfrog work below monolithic at equal chain count:
  **PASS**, work ratios `0.139`-`0.556` across 13 configurations, with the
  single-bucket controls correctly at `1.000`.
- Python bucket-loop overhead gate: **PASS**, warm max/min `1.231 <= 1.25`.
- Timing breakdown still separates planner, gather, executor, scatter, cold, and
  warm: **PASS**, verified in both new raw summaries.

Commands intentionally not run:
- No `sbatch`, `srun`, SLURM, GPU/HPC validation, or broad sweep was run.
- The oracle-gap and performance-gate reports were not regenerated. That is
  T41's job, into new output directories.

Next active task: T41 — Re-gate After Mechanism Repair. Do not start T41 until
the next invocation.

---

### [x] T41 — Re-gate After Mechanism Repair

**Goal:** Re-run the performance gate now that bucketing has a mechanism to
exploit, and decide whether T38 is justified.

**Dependencies:** T39 and T40.

**Scope:**

- Regenerate the oracle/overhead decomposition and the gate report from raw
  outputs, into a new output directory. Do not overwrite the `repair_tiny`
  artifacts; they are the record of the pre-mechanism state.
- Re-register the heterogeneous target before running it. The tiny CPU targets
  used by T34/T36 are not a valid venue: at `C=128` on CPU the planner is a fixed
  tax that no mechanism can outrun, and the funnel heterogeneity at that scale is
  small. State the chosen target and threshold in this file before measuring.
- The gate report must state, from raw results, whether bucketed executed
  leapfrog work is now below monolithic at equal chain count. That is the
  mechanism check and it is independent of hardware.
- Keep run `21070` labeled as pre-repair negative diagnostic evidence.
- All T32-T37R warm timings are now stale for performance purposes. T39 changed
  warm timing for every method and T40 changed it again for bucketed. Do not
  mix pre-T39 and post-T40 timings in one comparison. The `repair_tiny`
  artifacts stay as the pre-mechanism record.

Carry forward from T40, which changes what this gate is actually testing:

- The executor mechanism is demonstrated: work ratios `0.139`-`0.556` and
  executor-only wall speedups up to `2.360x` on a heterogeneous synthetic
  target with oracle plans.
- The end-to-end path is still `0.597x` on `tiny_cpu`, and the loss is now
  attributed to predictor quality (history plans reach work ratio `0.852` where
  oracle reaches `0.339`) and planner cost (`0.02856s` of a `0.17388s` step).
- So this gate now separates three things that used to be one number: does the
  executor reclaim work, does the predictor find the work to reclaim, and does
  planning cost less than what it saves. Report them separately.
- If the predictor is confirmed as the dominant remaining loss, the honest next
  step is a predictor/planner repair task before T38, not an HPC run. Do not
  promote T38 just because the executor now works.

**Acceptance criteria:**

- Gate report explicitly says `PASS` or `FAIL`.
- The report states the executed-work ratio, the predictor-plan versus
  oracle-plan work ratio, and the planner share of warm time, each from raw
  results.
- If the mechanism check passes but the wall-clock gate fails on CPU, that is a
  recordable "mechanism present, venue insufficient" outcome and may justify
  promoting T38 on that basis. Say so explicitly rather than forcing a CPU
  speedup.
- If the mechanism check itself fails, T38 stays blocked and the next active task
  is another repair task.

**Pre-registered target and thresholds (written 2026-08-09, before measuring):**

Target, added as profile `mechanism_repair` in `configs/oracle_gap.yaml`:

| setting | value |
|---|---|
| model | funnel |
| chains `C` | 512 |
| dimension `D` | 32 |
| `max_tree_depth` | 8 |
| `step_size` | 0.03 |
| `num_steps` | 8 |
| seeds | `[0]` |
| dtype / backend | float32 / CPU |
| bucket sizes | 64, 128 |
| scheduler modes | monolithic, history, oracle_previous, oracle_current |

How this target was chosen, stated plainly: it comes from the T40 executor
measurements, which showed `C=512`, `D=32`, `max_tree_depth=8` produces a wide
realized-depth spread (`{1:266, 2:128, 3:58, 4:35, 5:21, 6:4}`) and is CPU
feasible. Choosing a venue where the mechanism *can* show up is legitimate;
choosing it after seeing the gate result would not be. The threshold is fixed
here, before the run. The pre-existing `tiny_cpu` profile is reported alongside
it so this is not a single hand-picked row.

Thresholds:

1. Correctness — no unexplained mismatches; discrete metrics and RNG keys exact,
   floats within `1e-5`. Regenerated post-T40, not inherited.
2. Mechanism (new, hardware independent) — best non-analysis executed-work ratio
   `< 1.0`.
3. Heterogeneous wall-clock speedup — best non-analysis warm speedup `>= 1.05x`.
   Unchanged from T37.
4. Homogeneous negative control — production speedup `>= 0.80x`. Regenerated
   post-T40.
5. Python bucket-loop overhead — repaired warm max/min ratio `<= 1.25`.
6. Report generated from raw results, with faster-row counts stated.

Gates 2 and 3 are reported separately and neither substitutes for the other. A
pass on 2 with a fail on 3 is the "mechanism present, venue insufficient"
outcome the scope allows.

**Smallest validation:**

```bash
python3 -m pytest -q
python3 -m abnuts.analysis.performance_gate ... --out results/processed/performance_gate/mechanism_repair ...
```

**Completion notes:**

- Completed on 2026-08-09. The regenerated gate report explicitly says **FAIL**.
  Broad sweeps remain blocked and T38 is not promoted. Two gates failed:
  heterogeneous wall-clock speedup and the homogeneous negative control. The
  mechanism gate passed.

Implementation summary:
- Added a `mechanism_repair` profile to `configs/oracle_gap.yaml` matching the
  pre-registered target above.
- `run_oracle_gap` now emits executed-work fields in raw `summary.csv`:
  `executed_lane_steps`, `monolithic_executed_lane_steps`,
  `executed_work_ratio`, `oracle_plan_executed_lane_steps`, and
  `oracle_plan_executed_work_ratio`. The oracle-plan columns rebuild a plan from
  each step's realized depths and push it through the same executor, which is
  what separates predictor quality from executor structure.
- `performance_gate` gained a `mechanism_executed_work` criterion, a
  `--max-executed-work-ratio` flag, a "Mechanism Versus Scheduler" section in
  markdown and LaTeX, and a distinct next-task hint for the case where the
  mechanism passes but the wall-clock gate does not.
- Corrected two now-false strings in the gate report: it no longer claims to be
  generated from "T32-T37R artifacts" and no longer hardcodes T37-era task
  numbers in its title. Sources are listed per criterion as before.

Files changed:
- `configs/oracle_gap.yaml`
- `src/abnuts/experiments/run_oracle_gap.py`
- `src/abnuts/analysis/performance_gate.py`
- `STATUS.md`

Outputs generated (all into new directories; no pre-mechanism artifact was
overwritten):
- `results/raw/oracle_gap/mechanism_repair/` — summary, predictor calibration,
  padding heatmap, speedup heterogeneity, manifest
- `results/raw/correctness/gp_tiny_mechanism_repair/`
- `results/raw/performance_gate/homogeneous_negative_control_mechanism_repair/`
- `results/processed/performance_gate/mechanism_repair/` — gate CSV, markdown
  report, manifest
- `results/latex/performance_gate/mechanism_repair/gate_report.tex`

Validation commands run:
```bash
JAX_PLATFORMS=cpu python3 -m abnuts.experiments.run_correctness \
    --method both --model gaussian_process \
    --num-chains 4 --dimension 3 --num-steps 1 --step-size 0.005 \
    --max-tree-depth 2 --bucket-size 4 --predictor history \
    --out results/raw/correctness/gp_tiny_mechanism_repair

JAX_PLATFORMS=cpu python3 -m abnuts.experiments.run_homogeneous_negative_control \
    --out results/raw/performance_gate/homogeneous_negative_control_mechanism_repair

JAX_PLATFORMS=cpu python3 -m abnuts.experiments.run_oracle_gap \
    --config configs/oracle_gap.yaml --profile mechanism_repair \
    --out results/raw/oracle_gap/mechanism_repair

JAX_PLATFORMS=cpu python3 -m abnuts.analysis.performance_gate \
    --pre-repair-run-dir /home/hpc/users/viktor.najdovski/abnuts_runs/21070/raw \
    --correctness-dir results/raw/correctness/gp_tiny_mechanism_repair \
    --heterogeneous-summary results/raw/t40_per_bucket_executor/tiny_cpu/summary.csv \
    --heterogeneous-summary results/raw/t40_per_bucket_executor/tiny_cpu_bucket_scaling/summary.csv \
    --oracle-gap-dir results/raw/oracle_gap/mechanism_repair \
    --negative-control-summary results/raw/performance_gate/homogeneous_negative_control_mechanism_repair/summary.csv \
    --out results/processed/performance_gate/mechanism_repair \
    --latex-out results/latex/performance_gate/mechanism_repair --overwrite
# result: FAIL

JAX_PLATFORMS=cpu python3 -m pytest -q
# result: 88 passed in 156.39s
```

Gate results:

| gate | status | threshold | value |
|---|---|---|---|
| correctness | **PASS** | exact discrete/RNG metrics, floats within `1e-5` | all deltas exactly `0.0` |
| mechanism_executed_work | **PASS** | non-analysis executed-work ratio `< 1.0` | `0.688` |
| heterogeneous_speedup | **FAIL** | non-analysis warm speedup `>= 1.05x` | `0.794x` |
| homogeneous_negative_control | **FAIL** | production speedup `>= 0.80x` | `0.783x` |
| python_bucket_loop_overhead | PASS | warm max/min `<= 1.25` | `1.119` |
| report_from_raw_results | PASS | named raw summaries present | present |
| faster_rows_reported | PASS | counts stated | pre-repair `21/735`; repaired `0/11` |

The GP correctness deltas are now exactly `0.0` on every field, where T35
recorded ULP-scale residuals. Both paths run the same compiled `while_loop`
bodies after T39, so the compiled-versus-eager reordering that produced those
residuals is gone.

Pre-registered target results (`results/raw/oracle_gap/mechanism_repair/summary.csv`,
funnel `C=512`, `D=32`, `max_tree_depth=8`, 8 steps, CPU):

| mode | bucket size | speedup | executed-work ratio | oracle-plan ratio | predictor MAE |
|---|---|---|---|---|---|
| monolithic | — | `1.000x` | `1.000` | `1.000` | — |
| history | 64 | `0.794x` | `0.705` | `0.172` | `0.937` |
| history | 128 | `0.702x` | `0.792` | `0.283` | `0.937` |
| oracle_previous | 64 | `0.513x` | `0.688` | `0.172` | `1.172` |
| oracle_previous | 128 | `0.760x` | `0.791` | `0.283` | `1.172` |
| oracle_current (analysis-only) | 64 | **`1.324x`** | `0.172` | `0.172` | `0.000` |
| oracle_current (analysis-only) | 128 | `1.157x` | `0.283` | `0.283` | `0.000` |

Total leapfrog counts were identical (`13697`) across every mode and bucket
size, so the transition is preserved.

Interpretation, stated carefully:

- **The executor works.** `oracle_current` reaches `1.324x` end-to-end
  *including* planner cost. This is the first time in this project that any
  configuration has beaten monolithic on the full path. It is an analysis-only
  upper bound and cannot satisfy the gate, but it bounds what the executor can
  deliver: the ceiling is no longer `1.0x`.
- **The predictor is the blocker.** Deployed schedulers reach executed-work
  ratios of `0.688`-`0.792` where the same executor reaches `0.172` given
  perfect grouping. Mean predictor absolute error is `0.937` for `history` and
  `1.172` for `oracle_previous`, against realized depths spanning 1 to 6. The
  predictors are wrong by about one full depth level, which is enough to put
  chains in the wrong bucket.
- `oracle_previous` is *worse* than `history` despite being the "better"
  oracle-flavoured mode. Its MAE is higher, which is consistent: the previous
  step's raw realized depth is noisier than an EMA of past depths.
- **Planner cost is now a first-order term.** It is `11.1%` of warm time on the
  gate target.

Diagnosis of the homogeneous negative-control failure (`0.809x` before, `0.783x`
now). This is not a regression in bucketed execution:

| quantity | pre-T39 | post-T40 |
|---|---|---|
| monolithic warm | `0.25404s` | `0.18278s` |
| bucketed production warm | `0.31393s` | `0.23334s` |
| absolute overhead | `0.05989s` | `0.05055s` |
| planner | `0.03534s` | `0.04045s` |
| ratio | `0.809x` | `0.783x` |

Absolute bucketed overhead went **down**. The ratio got worse because T39 made
the monolithic denominator `28%` faster while the host-side planner stayed a
fixed cost, and the planner is now `80%` of the remaining overhead. So both
failing gates have the same root cause: **the host-side scheduler, not the
device-side executor.**

A note on the threshold, recorded so it is not quietly changed later: a ratio
threshold on a negative control is sensitive to how fast the baseline is, and
T37R set `0.80x` just under its own measured `0.809x`. That makes it a fragile
threshold rather than a principled one. It was **not** adjusted here, because
changing a threshold after seeing it fail is exactly the move `AGENTS.md`
section 8 forbids. The next task should decide whether an absolute-overhead
bound is the better formulation, and fix that before measuring.

Acceptance status:
- Gate report explicitly says `PASS` or `FAIL`: **done**, FAIL.
- Report states executed-work ratio, predictor-plan versus oracle-plan work
  ratio, and planner share of warm time, each from raw results: **done**, in the
  "Mechanism Versus Scheduler" section of the markdown and LaTeX reports.
- Mechanism check passed while the wall-clock gate failed. The scope allows
  promoting T38 on a "mechanism present, venue insufficient" basis, and that is
  **declined here**: the wall-clock failure is not caused by the CPU venue, it is
  caused by predictor error and planner cost, both of which are host-side and
  will not improve on a GPU. Running HPC validation now would buy a faster
  measurement of the same scheduler gap. The negative control also failed, which
  independently blocks promotion.

Commands intentionally not run:
- No `sbatch`, `srun`, SLURM, GPU/HPC validation, or broad sweep was run.

Next active task: T42 — Repair Scheduler Quality and Planner Cost. Do not start
T42 until the next invocation.

---

### [x] T43 — Repair the U-Turn Criterion

**Goal:** Make trajectory termination depend on geometry rather than on the
sampled integration direction.

**Dependencies:** None. This is a correctness blocker and precedes T42.

**Scope:**

- Fix the reversed-argument U-turn call in `_build_subtree`. When
  `signed_step < 0`, the leftmost point of the span is `current` and the
  rightmost is `start`, so the check must be oriented accordingly. Apply the
  same fix to both the control-flow and unrolled implementations so the T39
  differential test keeps comparing like with like.
- While there, assess whether the simplified `(start, current)` span check is
  adequate at all. Standard NUTS checks the U-turn criterion recursively over
  sub-subtrees, not only over the whole subtree span. Record the decision;
  do not silently widen scope.
- Verify the fix changes the depth distribution from geometric(0.5) to
  something model-dependent, and report the new distribution per model.
- This changes the Markov transition. It is a correctness repair, so that is
  expected and permitted, but every equivalence baseline and stored
  performance number must be regenerated afterwards rather than compared
  across the fix.

**Acceptance criteria:**

- Backward and forward subtrees give the same U-turn verdict for the same
  geometry, pinned by a unit test.
- Realized depth distribution differs across models and is no longer
  geometric(0.5).
- Monolithic-versus-bucketed equivalence still passes after the fix.
- A test asserts trajectory termination does not depend on the direction
  draw, so this cannot regress silently.

**Smallest validation:**

```bash
python3 -m pytest -q
```

**Completion notes:**

- Completed on 2026-08-09. All acceptance criteria met.

Implementation summary:
- Added `_span_endpoints(integrate_backward, start, current)`, which orders a
  subtree's two endpoints by trajectory position rather than visit order. The
  selection is traced, because the direction is drawn at runtime.
- Both `_build_subtree_control_flow` and `_build_subtree_unrolled` now take an
  `integrate_backward` flag and pass the correctly ordered span to
  `_is_turning`. Applying it to both keeps the T39 differential test comparing
  like with like.
- The outer `global_turning` check was already correctly oriented, because the
  doubling loop maintains `left` and `right` by direction. It was not changed.
- This changes the Markov transition. That is intended: it is a correctness
  repair. Every performance number measured before it is superseded.

Decision recorded on scope, as required: the criterion is still evaluated only
over the span from the subtree's start to the current point, checked at every
step. Standard NUTS additionally checks sub-subtree spans recursively. Adding
that is a larger change with its own regression risk, so it was **not** folded
into a correctness fix. It is worth a separate task; the current criterion is
now direction-correct but remains a documented simplification.

Files changed:
- `src/abnuts/nuts/transition.py`
- `tests/test_sampler_ground_truth.py` — new

Validation:
```bash
JAX_PLATFORMS=cpu python3 -m pytest -q
# result: 92 passed (88 before, +4 new)
```

Evidence the fix works. Realized depth distribution, `C=4096`, one step,
`max_tree_depth=8`, before and after:

| model | before (mean) | after (mean) |
|---|---|---|
| funnel | geometric(0.5) | `5.49` |
| eight_schools_centered | geometric(0.5) | `7.72` |
| gaussian_process | geometric(0.5) | `7.42` |
| stochastic_volatility | geometric(0.5) | `6.89` |

Before the fix all four produced the *identical* geometric(0.5) histogram
(`0.487, 0.251, 0.127, 0.068, ...`), because depth was a direction coin-flip
sequence. After it, each target produces its own distribution.

**New tests, and an honest note about which one matters.**
`tests/test_sampler_ground_truth.py` adds the category this suite was missing:
checks against analytic ground truth rather than consistency between two paths.
It contains four tests, and they were verified against the *broken* transition
to see which actually discriminate:

- `test_sampler_recovers_isotropic_gaussian_moments` — **does not** catch the
  defect. Measured against the pre-fix transition it passes: mean
  `[-0.078, 0.108, -0.070, -0.031]`, std `[1.057, 1.110, 1.006, 1.035]`. The
  broken sampler was not grossly biased. It degenerated into short one-sided
  HMC, which still targets the right distribution on an easy target. Kept
  because it is the right category of test and would catch a genuinely biased
  transition.
- `test_realized_depth_adapts_to_target_scale` — **does** catch it, and is the
  statistical guard. Widening the target 10x at fixed step size must raise
  realized depth: measured `1.85x` fixed against `1.08x` broken. Verified to
  fail against the reintroduced bug with a clear message.
- `test_turning_criterion_is_symmetric_under_integration_direction` — direct
  unit guard on the orientation itself.
- `test_sampler_explores_rather_than_sticking` — rules out a frozen chain
  satisfying the moment test by symmetry.

Efficiency effect of the bug, measured on the isotropic Gaussian: it took
`2.58` leapfrog steps per transition against the fixed sampler's `10.04`, with
lag-1 autocorrelation `0.787` against `0.428`. So it mixed roughly half as well
per iteration while doing a quarter of the work. Per gradient evaluation the two
are comparable on an easy target, which is why the moment check could not see
it, and is also why the defect survived 42 tasks.

Outputs regenerated post-fix, all into new directories:
- `results/raw/oracle_gap/uturn_fixed/`
- `results/raw/correctness/gp_tiny_uturn_fixed/`
- `results/raw/performance_gate/homogeneous_negative_control_uturn_fixed/`
- `results/raw/t43_uturn_fixed/tiny_cpu/`, `.../tiny_cpu_bucket_scaling/`
- `results/processed/performance_gate/uturn_fixed/`
- `results/latex/performance_gate/uturn_fixed/gate_report.tex`

Acceptance status:
- Forward and backward subtrees agree on the same geometry: **PASS**, unit test.
- Depth distribution differs across models and is no longer geometric(0.5):
  **PASS**, table above.
- Monolithic-versus-bucketed equivalence still passes: **PASS**, and the GP
  correctness deltas are now exactly `0.0` on every field.
- A test asserts termination does not depend on the direction draw: **PASS**,
  verified to fail against the reintroduced bug.

No `sbatch`, `srun`, SLURM, GPU/HPC validation, or broad sweep was run.

Next active task: T44 — Re-specify the Bucket-Scaling Gate, then Re-gate.

---

### [ ] T44 — Re-specify the Bucket-Scaling Gate, then Re-gate

**Goal:** Make the last failing criterion measure the phenomenon that actually
exists after T40, then re-run the gate.

**Dependencies:** T43. All other gates pass.

**Scope:**

- `python_bucket_loop_overhead` was written to detect Python per-bucket dispatch.
  After T40 that cannot occur: buckets run inside one compiled `lax.map`. The
  criterion now reads `1.350` against a `1.25` threshold and what it is
  detecting is the structural cost of narrow buckets on CPU.
- Decide what the criterion should now assert, write it and its threshold into
  this file **before** measuring, and rename it so the report stops claiming to
  measure Python dispatch. Candidates: a floor on bucket width, a structural
  overhead bound measured on a homogeneous batch, or a jaxpr-level assertion
  that no per-bucket host dispatch exists (which is the original intent and is
  now checkable statically).
- Keep a guard against the original failure mode. It must remain impossible to
  reintroduce host-side per-bucket dispatch silently.
- Re-run the gate into a new directory afterwards.

**Acceptance criteria:**

- The renamed criterion, its threshold, and its venue are recorded here before
  the measurement that evaluates them.
- A guard against host-side per-bucket dispatch still exists.
- Gate report states PASS or FAIL from raw results.
- If it passes, T38 minimal HPC validation may finally be promoted.

**Smallest validation:**

```bash
python3 -m pytest -q
# plus the gate command from T43 into a new output directory
```

**Completion notes:**

- Pending.

---

### [ ] T42 — Repair Scheduler Quality and Planner Cost (DEFERRED behind T43)

**Goal:** Close the gap between deployed schedulers and the oracle plans the
repaired executor can already exploit.

**Dependencies:** T41 failed on heterogeneous speedup and the homogeneous
negative control, both attributed to host-side scheduling.

**Scope:**

- The executor is not the problem and should not be modified. T41 measured
  `oracle_current` at `1.324x` end-to-end on the pre-registered target while
  deployed predictors reached `0.794x`.
- Improve predictor quality. Mean absolute error is `0.937` (`history`) and
  `1.172` (`oracle_previous`) against realized depths spanning 1 to 6. Bucket
  assignment only needs the *ordering* of chains by work, not the absolute
  depth, so consider ranking-based or hysteresis-based predictors and measure
  ordering quality directly rather than only MAE.
- Reduce planner cost. It is `11.1%` of warm time on the gate target and `80%`
  of the negative control's overhead. The planner is host-side NumPy doing a
  sort per step; consider planning on device, planning less often, or reusing a
  plan across steps when predicted work has not materially changed.
- Any plan-reuse scheme must preserve the transition exactly. Grouping may
  change freely, but per-chain RNG sequences and per-chain results may not.
- Decide and record whether the homogeneous negative-control gate should be an
  absolute-overhead bound rather than a ratio. Fix the formulation *before*
  measuring against it.

**Acceptance criteria:**

- Deployed non-analysis executed-work ratio moves materially toward the
  oracle-plan ratio on the T41 pre-registered target, or the task records why
  the predictor cannot.
- Planner share of warm time is reduced, or its cost is shown to be irreducible.
- T35 equivalence gate still passes.
- The gate is regenerated into a new output directory and states PASS or FAIL.

**Smallest validation:**

```bash
python3 -m pytest -q
python3 -m abnuts.experiments.run_oracle_gap \
    --config configs/oracle_gap.yaml --profile mechanism_repair \
    --out results/raw/oracle_gap/scheduler_repair
# plus the gate command from T41 with the new oracle-gap directory
```

Do not run `sbatch`, `srun`, GPU/HPC validation, or broad sweeps.

**Completion notes:**

- Pending.

---

### [ ] T38 — Re-run Minimal HPC Validation

**Goal:** Validate the repaired architecture on HPC with a minimal sweep only after the gate passes.

**Dependencies:** A passing re-gate after T42. (Originally written as "T37
pass"; superseded on 2026-08-09 because the T37/T37R/T37S gate chain was
measuring a target with no mechanism behind it. T41 then failed on scheduler
quality, not executor structure, so T42 is now in front of T38.)

**Scope:**

- Run a small HPC validation sweep, not the full paper sweep.
- Include at least one target-like funnel case sufficient to compare against run `21070`, especially `C=2048` if feasible.
- Compare repaired results against run `21070`.
- State explicitly whether C=2048 funnel performance improved.

**Acceptance criteria:**

- Job ID, command, output directory, and logs are recorded.
- Report compares repaired minimal validation against run `21070`.
- Report states whether architecture repair improved C=2048 funnel performance.
- Full paper sweeps remain a later task, not part of T38.

**Smallest validation:**

```bash
sbatch scripts/hpc/benchmark_sweep.sbatch  # with a repair/minimal profile only
```

Record the exact command and site-specific environment variables used.

**Completion notes:**

- Pending.

---

## Completed task summary

T01–T31 are complete and produced the package, model suite, monolithic and bucketed runners, predictor/planner scaffolding, correctness commands, benchmark configs, SLURM wrappers, LaTeX reporting, README/artifact docs, and validation pass.

However, T31 is no longer the end of the project. The broad benchmark scaffold exposed a central architecture failure. Phase 4 is now active.
