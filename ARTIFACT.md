# Artifact Checklist

This checklist describes how to reproduce the ABNUTS software artifact without
using private paths. Commands assume the repository root is the current working
directory.

## Environment

Required local software:

- Python `>=3.10,<3.13`
- A POSIX shell for wrapper scripts
- `pip`

Optional HPC software:

- Singularity or Apptainer
- SLURM
- NVIDIA GPU drivers for GPU runs

Install local dependencies:

```bash
python -m pip install -e '.[dev]'
```

The container definition is `singularity/abnuts.def`. The expected built image
path is `images/abnuts.sif`.

## Hardware

CPU tests and tiny smoke profiles should run on a login node, workstation, or
small CI runner. Paper-scale benchmark profiles are intended for an accelerator
node and should be submitted through SLURM or an equivalent batch system.

The code records available runtime metadata in manifests where supported. For
HPC jobs, keep the SLURM logs with the raw result directory so hardware and
environment output are preserved.

## Minimal CPU Reproduction

Run the unit tests:

```bash
python -m pytest -q
```

Run the local CPU smoke command:

```bash
bash scripts/local/smoke_cpu.sh
```

Expected output:

```text
${SCRATCH:-$HOME}/abnuts_runs/${SLURM_JOB_ID:-local}/smoke_cpu/manifest.json
```

## Tiny Raw Results

The following commands exercise representative raw-output schemas on CPU-sized
profiles:

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

python -m abnuts.experiments.run_sbc \
  --config configs/sbc.yaml \
  --profile tiny \
  --out results/raw/sbc/tiny
```

Use `--overwrite` only when intentionally replacing an existing raw output.

## Container Reproduction

Build the image:

```bash
bash scripts/hpc/build_image.sh
```

Site-specific example using Apptainer:

```bash
MODULES_TO_LOAD="apptainer" \
SINGULARITY_BIN="apptainer" \
bash scripts/hpc/build_image.sh
```

Expected output:

```text
images/abnuts.sif
```

Run a GPU smoke through SLURM after editing account, partition, and resource
directives:

```bash
sbatch scripts/hpc/smoke_gpu.sbatch
```

Expected output directory:

```text
${SCRATCH:-$HOME}/abnuts_runs/<job-id>/smoke_gpu
```

Expected logs:

```text
logs/abnuts-smoke-<job-id>.out
logs/abnuts-smoke-<job-id>.err
```

## Paper-Scale Runs

The main SLURM wrappers are:

```bash
sbatch scripts/hpc/benchmark_sweep.sbatch
sbatch scripts/hpc/correctness_sweep.sbatch
sbatch scripts/hpc/oracle_gap.sbatch
sbatch scripts/hpc/profile_nsight.sbatch
```

Before submission, set site-specific `#SBATCH` fields and confirm:

- `PROJECT_ROOT` points to this repository.
- `IMAGE` points to `images/abnuts.sif` or another validated image.
- `RUNROOT` points to durable scratch storage.
- `MODULES_TO_LOAD` and `SINGULARITY_BIN` match the cluster.

The sweep scripts support task introspection and dry runs:

```bash
ABNUTS_LIST_TASKS=1 bash scripts/hpc/benchmark_sweep.sbatch
ABNUTS_DRY_RUN=1 ABNUTS_ARRAY_TASK_ID=0 bash scripts/hpc/benchmark_sweep.sbatch
```

## Reproduce Tables and Figures

Generate a single-experiment LaTeX report:

```bash
python -m abnuts.analysis.report \
  --input results/raw/funnel_ablation/tiny \
  --out results/latex/funnel_ablation/tiny
```

Generate the aggregate artifact bundle:

```bash
python -m abnuts.analysis.report \
  --input results/raw \
  --out results/latex/all \
  --allow-missing
```

Expected aggregate outputs:

- `results/latex/all/manifest.json`
- `results/latex/all/results_summary.tex`
- `results/latex/all/sensitivity_summary.tex`
- `results/latex/all/macros/result_macros.tex`
- `results/latex/all/tables/*.tex`
- `results/latex/all/figures/*.tex`
- `results/processed/all/*.csv`

`TODO_RESULT` markers mean the artifact was generated from incomplete tiny or
partial raw results and should not be treated as a paper-scale claim.

## Result Directory Map

- `results/raw/<experiment>/<run-id>/`: immutable raw experiment outputs.
- `results/processed/all/`: CSV files collated by aggregate analysis.
- `results/latex/<experiment>/<run-id>/`: single-experiment LaTeX outputs.
- `results/latex/all/`: aggregate LaTeX bundle for the paper.
- `logs/`: SLURM stdout/stderr logs.

Canonical paper artifacts are LaTeX-first. Do not treat generated preview images
or PDFs as source artifacts.

## Verification Checklist

- Unit tests pass with `python -m pytest -q`.
- CPU smoke writes a `manifest.json`.
- Container image exists at `images/abnuts.sif` before GPU/HPC validation.
- SLURM logs record host, image, run root, and output directory.
- Raw results are preserved unless `--overwrite` was explicitly used.
- Aggregate report generation completes.
- Oracle-current rows are labeled as analysis-only upper bounds.
- Missing paper-scale results remain marked as `TODO_RESULT`.
