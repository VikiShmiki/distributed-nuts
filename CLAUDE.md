# AGENTS.md — Asynchronous Bucketed NUTS Research Implementation

## 0. Read this first

This repository implements **Asynchronous Bucketed NUTS** in JAX for research-grade benchmarking on CPU and accelerator/HPC systems.

The current project state is **not complete**. Tasks T01–T31 built a broad implementation and benchmark scaffold, but HPC run `21070` showed that the present bucketed path does **not** yet realize the intended device-side fixed-shape executor architecture. The immediate priority is therefore **performance architecture repair**, not broader benchmarking.

When invoked, every coding agent must:

1. Read `AGENTS.md`.
2. Read `RESEARCH.md`.
3. Read `STATUS.md`.
4. Identify the single active task in `STATUS.md`.
5. Implement only that task.
6. Run the smallest validation listed for that task.
7. Fix only failures caused by that task.
8. Update `STATUS.md` with files changed, validation run, results, and the next active task.
9. Stop.

`STATUS.md` is the authoritative task board. Do not infer or invent tasks from old completion notes.

---

## 1. Mission

Build a clean, reproducible, HPC-ready JAX codebase for testing whether bucketing can make multi-chain NUTS faster on accelerators **without changing the Markov transition**.

Core idea:

> Multi-chain NUTS wastes accelerator work when chains realize different tree depths. A scheduler predicts per-chain work, groups chains with similar expected work into fixed-shape buckets, pads/masks bucket lanes, and executes the same underlying NUTS transition with less straggler waste.

Important distinction:

- The draft paper describes a target architecture: **host-side planning plus a device-side fixed-shape/single-JIT executor**.
- The current codebase has not yet reproduced that architecture at paper strength.
- The current evidence from HPC run `21070` is mostly negative and must be treated as a diagnosis, not a publishable speedup claim.

The codebase should eventually answer:

1. Where does bucketing help?
2. Why does it help or fail?
3. Does it preserve the NUTS transition exactly or within documented strict tolerance?
4. How much slowdown comes from predictor error, padding, gather/scatter, compile cost, and executor overhead?
5. Does the repaired executor improve target-like cases such as `C=2048` funnel runs?

---

## 2. Scientific invariants

These override speed, convenience, task scope, and old documentation.

### 2.1 Transition preservation

Bucketed NUTS is a scheduling/execution layer around the same per-chain transition as the monolithic reference.

Allowed differences:

- execution order,
- grouping,
- fixed-shape padding,
- masked gather/scatter,
- instrumentation,
- compile/executor organization that preserves semantics.

Forbidden differences:

- changing the per-chain RNG sequence in equivalence tests,
- changing the leapfrog integrator,
- changing the U-turn criterion,
- changing divergence or max-depth logic,
- changing adaptation logic unless a task explicitly isolates it,
- allowing padded lanes to influence real chain states or metrics,
- silently comparing different dtype/precision settings.

### 2.2 Correctness gates

For every executor rewrite or benchmark-family addition, include deterministic monolithic-vs-bucketed checks with identical per-chain RNG keys.

Minimum checks:

- positions,
- RNG-derived transition metrics,
- realized tree depth,
- leapfrog count,
- divergence flag,
- max-depth flag,
- acceptance statistic or documented strict-tolerance equivalent,
- padded-lane noninterference.

If bitwise equality is impossible because JAX changes operation ordering, use strict numerical tolerances and document the exact reason in the test or report. Do not weaken tests merely to pass.

### 2.3 Timing correctness

All JAX timing must call `block_until_ready()` or an equivalent tree-blocking helper before reading wall time.

When relevant, separate:

- first compile / cold start,
- warm iteration time,
- host planning time,
- gather time,
- executor/transition time,
- scatter time,
- total wall time.

Never report asynchronous dispatch time as execution time.

### 2.4 Performance architecture invariant

Bucketed execution may not be considered complete if the hot path loops over buckets in Python and creates one accelerator dispatch per bucket **unless that overhead is explicitly measured, justified, and shown not to dominate**.

A valid repaired bucketed executor must use a JAX-native or compiled fixed-shape strategy that avoids host-side per-bucket dispatch dominating warm runtime. Acceptable designs include:

- one cached compiled executor per canonical bucket size,
- one compiled padded rectangular executor using `lax.scan`, `lax.map`, `vmap`, or another JAX-native control-flow strategy,
- another fixed-shape compiled design that preserves the same NUTS transition and measures gather/executor/scatter overhead separately.

Host-side planning is allowed. Python loops over buckets in the warm transition hot path are not acceptable as the final architecture unless the performance report proves they are harmless.

### 2.5 LaTeX-first artifacts

Paper artifacts must be reproducible from raw outputs and LaTeX-first.

Canonical outputs:

- raw results: `.csv`, `.json`, `.jsonl`, copied configs, manifests,
- processed results: `.csv`, `.json`, `.jsonl`,
- tables: `.tex`,
- figures: `.tex` using PGFPlots/TikZ,
- macros: `.tex`.

Preview PNG/PDF files are allowed only as debugging artifacts. They are not canonical paper artifacts.

---

## 3. Current evidence lock from run 21070

The following must be reflected honestly in reports and research planning:

- Broad sweep outputs mostly completed, except `hierarchical_logistic` initially.
- Only `21/690` completed bucketed rows were faster than monolithic, all in `chain_scaling`.
- Best observed speedup was `1.264x` at small chain count: `C=128`, `D=128`, history predictor, bucket size `512`.
- Target-like `C=2048` cases were much slower for bucketed execution.
- Best bucketed speedups by family were negative: `funnel_ablation` `0.632x`, `dimension_scaling` `0.309x`, `eight_schools_centered` `0.725x`, `eight_schools_noncentered` `0.530x`, `stochastic_volatility` `0.498x`, `gaussian_process` `0.512x`.
- `gaussian_process` showed bucketed-vs-monolithic leapfrog-count mismatches in many rows.
- The observed slowdown appears to scale with number of buckets, consistent with serial dispatch / host overhead.

No document, table, caption, README, or manuscript text may imply that this codebase has reproduced the draft `~1.28x` `C=2048` A100 funnel speedup until a repaired architecture does so from raw results.

---

## 4. Benchmark and reporting rules

Broad benchmark sweeps are blocked until the performance gates in `STATUS.md` pass.

Every benchmark summary must report:

- total monolithic rows,
- total bucketed rows,
- number and percentage of bucketed rows faster than monolithic,
- best/worst/median warm speedup,
- whether any transition-metric mismatches occurred,
- mismatch counts by model/family,
- timing methodology: cold/warm, blocking method, backend, hardware, JAX/jaxlib version,
- whether oracle-current rows are analysis-only upper bounds.

Any report using run `21070` must label it as **negative diagnostic evidence from the pre-repair executor**, not final paper evidence.

Do not run a full paper sweep as part of a small implementation task. Full or broad sweeps are allowed only when `STATUS.md` explicitly promotes such a task after the architecture gates pass.

---

## 5. Implementation discipline

### 5.1 Scope control

Implement only the active task. Do not perform opportunistic refactors, future tasks, paper-polish tasks, or broad benchmark additions.

### 5.2 Correctness before speed

A fast but semantically changed sampler is invalid. A correct but slow implementation is acceptable during repair, but it must be labeled honestly.

### 5.3 Preserve raw results

Never overwrite raw results unless an explicit `--overwrite` option is passed. Preserve configs, manifests, logs, and command lines.

### 5.4 Keep modules small

Prefer boring, typed, testable modules over clever abstractions. Avoid notebooks as source of truth.

### 5.5 Failure handling

If a command fails, record:

- command,
- working directory,
- backend/device,
- output directory,
- log path,
- short failure diagnosis,
- whether the failure blocks the active task.

Do not mark a task complete if its acceptance criteria failed.

---

## 6. HPC and long-run rules

Use Singularity/Apptainer + SLURM wrappers when tasks require accelerator validation.

For any SLURM or long run, record in `STATUS.md`:

- exact command,
- job ID or process ID,
- output directory,
- stdout/stderr logs,
- config file/profile,
- expected result files,
- whether the task is complete, running, or waiting.

If a required run is long, submit only the smallest task-relevant validation. Do not start broad sweeps unless the active task requires them.

---

## 7. Minimal commands to keep working

Local validation:

```bash
python -m pytest -q
bash scripts/local/smoke_cpu.sh
python -m abnuts.analysis.report --input results/raw --out results/latex/all --allow-missing
```

HPC/script validation:

```bash
bash -n scripts/hpc/build_image.sh
bash -n scripts/hpc/smoke_gpu.sbatch
bash -n scripts/hpc/benchmark_sweep.sbatch
bash -n scripts/hpc/correctness_sweep.sbatch
bash -n scripts/hpc/oracle_gap.sbatch
bash -n scripts/hpc/profile_nsight.sbatch
```

Task-specific commands in `STATUS.md` override this list.

---

## 8. Forbidden behaviors

Do not:

- claim paper-scale speedups from tiny, partial, or pre-repair diagnostic runs,
- treat old MC/HPC outputs as final if the protocol or executor changed,
- weaken correctness tests to hide mismatches,
- ignore Gaussian-process leapfrog-count mismatches,
- remove block-until-ready timing,
- hide Python dispatch overhead inside “bucketed runtime,”
- run broad sweeps before the performance gate,
- update `RESEARCH.md` to be more optimistic than raw results justify,
- leave `STATUS.md` with zero active tasks while architecture repair is incomplete.
