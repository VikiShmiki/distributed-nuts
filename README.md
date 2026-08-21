# Asynchronous Bucketed NUTS

This repository contains a research implementation of Asynchronous Bucketed NUTS
(ABNUTS): a scheduling layer for multi-chain NUTS that groups chains by predicted
work while preserving the underlying monolithic NUTS transition.

The implementation is designed for CPU smoke tests, GPU/HPC runs through
Singularity or Apptainer, and LaTeX-first paper artifacts generated from raw
CSV/JSON outputs.

## Repository Layout

- `configs/`: experiment profiles, with small CPU `tiny` profiles and larger
  HPC-oriented profiles.
- `scripts/local/`: local smoke wrappers.
- `scripts/hpc/`: Singularity/Apptainer and SLURM wrappers.
- `singularity/abnuts.def`: container definition.
- `src/abnuts/`: sampler, models, experiments, and analysis code.
- `tests/`: CPU-runnable pytest suite.
- `results/raw/`: raw experiment outputs.
- `results/processed/`: aggregate CSVs derived from raw outputs.
- `results/latex/`: canonical paper artifacts, including `.tex` tables,
  `.tex` PGFPlots/TikZ figures, and `.tex` macros.

Generated LaTeX and processed files should be regenerated from raw results when
possible. Raw results are never overwritten unless the relevant command is run
with `--overwrite`.

## Local Setup

From a fresh clone, create or activate a Python environment with Python
`>=3.10,<3.13`, then install the package and development dependencies:

```bash
python -m pip install -e '.[dev]'
```

If your shell only provides `python3`, use `python3` in place of `python` in the
commands below.

Run the unit tests:

```bash
python -m pytest -q
```

Run the local CPU smoke wrapper:

```bash
bash scripts/local/smoke_cpu.sh
```

By default this writes to:

```text
${SCRATCH:-$HOME}/abnuts_runs/${SLURM_JOB_ID:-local}/smoke_cpu
```

For an in-repository smoke output instead, run:

```bash
python -m abnuts.experiments.smoke \
  --config configs/smoke.yaml \
  --backend cpu \
  --out results/raw/smoke/local
```

## Common Local Experiment Commands

Tiny CPU profiles are intended to validate wiring, schemas, and artifact
generation. They are not paper-scale evidence.

```bash
python -m abnuts.experiments.run_benchmark \
  --config configs/funnel_ablation.yaml \
  --profile tiny \
  --out results/raw/funnel_ablation/tiny

python -m abnuts.experiments.run_correctness \
  --model funnel \
  --num-chains 8 \
  --dimension 4 \
  --num-steps 8 \
  --method both \
  --save-trace \
  --out results/raw/trace_qa/tiny

python -m abnuts.experiments.run_oracle_gap \
  --config configs/oracle_gap.yaml \
  --profile tiny \
  --out results/raw/oracle_gap/tiny
```

Generate LaTeX from one raw result directory:

```bash
python -m abnuts.analysis.report \
  --input results/raw/funnel_ablation/tiny \
  --out results/latex/funnel_ablation/tiny
```

Generate the aggregate LaTeX bundle from all available raw results:

```bash
python -m abnuts.analysis.report \
  --input results/raw \
  --out results/latex/all \
  --allow-missing
```

When `--allow-missing` is used, missing paper-scale evidence is marked with
`TODO_RESULT` instead of being silently filled in.

## Container Build

The Singularity/Apptainer definition file is:

```text
singularity/abnuts.def
```

The expected image path used by the HPC scripts is:

```text
images/abnuts.sif
```

Build it on a system with Singularity or Apptainer:

```bash
bash scripts/hpc/build_image.sh
```

Site-specific settings can be supplied through environment variables:

```bash
MODULES_TO_LOAD="apptainer" \
SINGULARITY_BIN="apptainer" \
bash scripts/hpc/build_image.sh
```

If `images/abnuts.sif` already exists, the HPC scripts will use it by default.
If it does not exist, container validation has not happened yet; use the local
CPU smoke path or build the image before submitting GPU jobs.

## SLURM Runs

Before submitting any `.sbatch` file, edit or override the site-specific
directives near the top of the script:

- `#SBATCH --partition=EDIT_ME`
- `#SBATCH --account=EDIT_ME`
- GPU, CPU, memory, and time limits as required by the cluster.

The scripts also accept these environment variables:

- `PROJECT_ROOT`: repository path to bind into the container as `/workspace`.
- `IMAGE`: `.sif` image path, defaulting to `images/abnuts.sif`.
- `RUNROOT`: host output root, defaulting to
  `${SCRATCH:-$HOME}/abnuts_runs/${SLURM_JOB_ID:-local}`.
- `MODULES_TO_LOAD`: optional module list for the local cluster.
- `SINGULARITY_BIN`: `singularity` or `apptainer`.

GPU smoke:

```bash
sbatch scripts/hpc/smoke_gpu.sbatch
```

Benchmark sweep:

```bash
sbatch scripts/hpc/benchmark_sweep.sbatch
```

Long-trace correctness:

```bash
sbatch scripts/hpc/correctness_sweep.sbatch
```

Oracle-gap sweep:

```bash
sbatch scripts/hpc/oracle_gap.sbatch
```

Useful dry-run/listing helpers:

```bash
ABNUTS_LIST_TASKS=1 bash scripts/hpc/benchmark_sweep.sbatch
ABNUTS_DRY_RUN=1 ABNUTS_ARRAY_TASK_ID=0 bash scripts/hpc/benchmark_sweep.sbatch
```

The SLURM scripts print the resolved image path, output directory, and log paths
at startup. Logs are written under `logs/` using the script's `#SBATCH --output`
and `#SBATCH --error` patterns.

## Results and Artifacts

Raw outputs use machine-readable files such as:

- `manifest.json`
- `summary.csv`
- `diagnostics.csv`
- `events.jsonl`
- `per_iteration.csv`
- `equivalence.json`
- `rank_histogram.csv`

Aggregate analysis writes:

- `results/processed/all/*.csv`
- `results/latex/all/results_summary.tex`
- `results/latex/all/macros/result_macros.tex`
- `results/latex/all/tables/*.tex`
- `results/latex/all/figures/*.tex`

LaTeX files include comment headers with the command and source file used to
generate them. Optional preview images are not canonical artifacts.

For a reviewer-facing checklist, see `ARTIFACT.md`.

## Troubleshooting

- `python: command not found`: use `python3`, or set `PYTHON_BIN=python3` for
  `scripts/local/smoke_cpu.sh`.
- Existing outputs are preserved: pass `--overwrite` only when intentionally
  replacing raw results.
- Missing image on HPC: build `images/abnuts.sif` with
  `scripts/hpc/build_image.sh` or set `IMAGE=/path/to/abnuts.sif`.
- `singularity` not found: set `SINGULARITY_BIN=apptainer` or load the site's
  container module with `MODULES_TO_LOAD`.
- No GPU visible: CPU smoke tests should still work. GPU scripts print
  `nvidia-smi` output when available and otherwise continue only for CPU-capable
  validation paths.
- Analysis fails with missing results: use `--allow-missing` for aggregate
  smoke reports, or run the missing experiment first.
