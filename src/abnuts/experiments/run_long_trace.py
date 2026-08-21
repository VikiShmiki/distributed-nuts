"""Run long-trace monolithic-vs-bucketed correctness checks."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr

from abnuts.analysis.diagnostics import DiagnosticsDependencyError
from abnuts.blocking import block_until_ready_tree
from abnuts.config import ConfigError, load_yaml_config
from abnuts.experiments.run_correctness import (
    _build_trace_diagnostics,
    _trace_qa_summary,
    write_run_outputs,
)
from abnuts.io import build_result_manifest, write_manifest
from abnuts.models import get_model
from abnuts.nuts.bucketed import BucketedRunResult, run_bucketed
from abnuts.nuts.monolithic import MonolithicRunResult, run_monolithic
from abnuts.nuts.predictors import PREDICTOR_MODES

MANIFEST_MATCH_KEYS = (
    "schema_version",
    "command",
    "config_sha256",
    "config_hash",
    "profile",
    "benchmark",
)


class LongTraceConfig(NamedTuple):
    """Resolved long-trace correctness configuration for one profile."""

    benchmark: str
    model: str
    backend: str
    dtype: str
    seed: int
    num_chains: int
    dimension: int
    model_config: dict[str, Any]
    num_steps: int
    step_size: float
    max_tree_depth: int
    bucket_size: int
    predictor: str
    predictor_beta: float


class LongTraceResult(NamedTuple):
    """Outputs from one long-trace correctness comparison."""

    monolithic: MonolithicRunResult
    bucketed: BucketedRunResult
    diagnostic_rows: tuple[Any, ...]
    trace_qa: dict[str, Any]
    model_metadata: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the long-trace runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu", "gpu"], help="Override configured backend.")
    parser.add_argument("--profile", default="tiny")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_config(
    raw_config: dict[str, Any],
    *,
    profile: str,
    backend_override: str | None = None,
) -> LongTraceConfig:
    """Resolve top-level defaults plus one named profile into a typed config."""
    profiles = raw_config.get("profiles", {})
    if not isinstance(profiles, Mapping):
        raise ValueError("config field 'profiles' must be a mapping")
    if profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise ValueError(
            f"profile {profile!r} not found in config. Available profiles: {available}"
        )
    profile_config = profiles[profile]
    if not isinstance(profile_config, Mapping):
        raise ValueError(f"profile {profile!r} must contain a mapping")

    merged = {key: value for key, value in raw_config.items() if key != "profiles"}
    merged.update(profile_config)
    if backend_override is not None:
        merged["backend"] = backend_override
    config = LongTraceConfig(
        benchmark=str(merged.get("benchmark", "long_trace")),
        model=str(merged.get("model", "funnel")),
        backend=str(merged.get("backend", "cpu")),
        dtype=str(merged.get("dtype", "float32")),
        seed=_as_int(merged.get("seed", 0), "seed"),
        num_chains=_as_positive_int(merged.get("num_chains", 4), "num_chains"),
        dimension=_as_positive_int(merged.get("dimension", 4), "dimension"),
        model_config=_as_model_config(merged.get("model_config", {})),
        num_steps=_as_positive_int(merged.get("num_steps", 12), "num_steps"),
        step_size=_as_positive_float(merged.get("step_size", 0.03), "step_size"),
        max_tree_depth=_as_positive_int(merged.get("max_tree_depth", 2), "max_tree_depth"),
        bucket_size=_as_positive_int(merged.get("bucket_size", 2), "bucket_size"),
        predictor=str(merged.get("predictor", "history")),
        predictor_beta=_as_probability(merged.get("predictor_beta", 0.9), "predictor_beta"),
    )
    validate_config(config)
    return config


def validate_config(config: LongTraceConfig) -> None:
    """Validate resolved long-trace settings."""
    if config.benchmark != "long_trace":
        raise ValueError(f"expected benchmark='long_trace', got {config.benchmark!r}")
    if config.backend not in {"cpu", "gpu"}:
        raise ValueError(f"run_long_trace supports backend='cpu' or 'gpu', got {config.backend!r}")
    if config.dtype != "float32":
        raise ValueError(f"run_long_trace currently supports dtype='float32', got {config.dtype!r}")
    if config.num_steps < 2:
        raise ValueError("long-trace diagnostics require num_steps >= 2")
    if config.predictor not in PREDICTOR_MODES:
        valid = ", ".join(sorted(PREDICTOR_MODES))
        raise ValueError(f"unsupported predictor {config.predictor!r}; expected one of {valid}")


def run_long_trace(config: LongTraceConfig) -> LongTraceResult:
    """Run monolithic and bucketed traces under identical initial positions and RNG."""
    model = get_model(
        config.model,
        dimension=config.dimension,
        model_config=config.model_config,
    )
    initial_positions = jnp.asarray(
        model.initial_position(
            key=config.seed,
            num_chains=config.num_chains,
            config=config.model_config,
        ),
        dtype=jnp.float32,
    )
    block_until_ready_tree(initial_positions)
    rng_key = jr.PRNGKey(config.seed)

    monolithic = run_monolithic(
        model,
        initial_positions,
        rng_key,
        num_steps=config.num_steps,
        step_size=config.step_size,
        max_tree_depth=config.max_tree_depth,
        dtype=jnp.float32,
    )
    bucketed = run_bucketed(
        model,
        initial_positions,
        rng_key,
        num_steps=config.num_steps,
        step_size=config.step_size,
        max_tree_depth=config.max_tree_depth,
        bucket_size=config.bucket_size,
        predictor=config.predictor,
        predictor_beta=config.predictor_beta,
        dtype=jnp.float32,
    )
    monolithic, bucketed = block_until_ready_tree((monolithic, bucketed))
    diagnostic_rows = _build_trace_diagnostics(monolithic, bucketed)
    trace_qa = _trace_qa_summary(monolithic, bucketed, diagnostic_rows)
    return LongTraceResult(
        monolithic=monolithic,
        bucketed=bucketed,
        diagnostic_rows=diagnostic_rows,
        trace_qa=trace_qa,
        model_metadata=model.metadata.as_dict(),
    )


def write_outputs(
    out_dir: Path,
    *,
    config_path: Path,
    profile: str,
    config: LongTraceConfig,
    result: LongTraceResult,
    overwrite: bool,
) -> None:
    """Write long-trace raw outputs and manifest."""
    resolved_config = _config_to_json(config, profile=profile)
    manifest = build_result_manifest(
        command="python -m abnuts.experiments.run_long_trace",
        config_path=config_path,
        output_dir=out_dir,
        config=resolved_config,
        extra={
            "profile": profile,
            "benchmark": config.benchmark,
            "model_metadata": result.model_metadata,
            "equivalence": result.trace_qa,
            "outputs": {
                "manifest_json": "manifest.json",
                "diagnostics_csv": "diagnostics.csv",
                "trace_qa_json": "trace_qa.json",
                "equivalence_json": "equivalence.json",
                "monolithic_positions_csv": "monolithic_positions.csv",
                "bucketed_positions_csv": "bucketed_positions.csv",
                "bucketed_bucket_plan_csv": "bucketed_bucket_plan.csv",
            },
        },
    )
    write_manifest(
        out_dir,
        manifest,
        overwrite=overwrite,
        comparable_keys=MANIFEST_MATCH_KEYS,
    )
    write_run_outputs(
        out_dir,
        method="both",
        monolithic_result=result.monolithic,
        bucketed_result=result.bucketed,
        diagnostic_rows=result.diagnostic_rows,
        trace_qa_summary=result.trace_qa,
        overwrite=overwrite,
    )


def _config_to_json(config: LongTraceConfig, *, profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "benchmark": config.benchmark,
        "model": config.model,
        "backend": config.backend,
        "dtype": config.dtype,
        "seed": config.seed,
        "num_chains": config.num_chains,
        "dimension": config.dimension,
        "model_config": config.model_config,
        "num_steps": config.num_steps,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "bucket_size": config.bucket_size,
        "predictor": config.predictor,
        "predictor_beta": config.predictor_beta,
    }


def _as_int(value: Any, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _as_positive_int(value: Any, name: str) -> int:
    parsed = _as_int(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed!r}")
    return parsed


def _as_positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive, got {parsed!r}")
    return parsed


def _as_probability(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {parsed!r}")
    return parsed


def _as_model_config(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("model_config must be a mapping")
    return {str(key): item for key, item in value.items()}


def main(argv: list[str] | None = None) -> int:
    """Run the long-trace correctness command."""
    args = parse_args(argv)
    backend = "unknown"
    try:
        raw_config = load_yaml_config(args.config)
        config = resolve_config(raw_config, profile=args.profile, backend_override=args.backend)
        backend = config.backend
        jax.config.update("jax_platform_name", config.backend)
        result = run_long_trace(config)
        write_outputs(
            args.out,
            config_path=args.config,
            profile=args.profile,
            config=config,
            result=result,
            overwrite=args.overwrite,
        )
    except (ConfigError, DiagnosticsDependencyError, OSError, ValueError) as exc:
        print(
            "Long-trace correctness run failed "
            f"(config={getattr(args, 'config', 'unknown')}, "
            f"profile={getattr(args, 'profile', 'unknown')}, "
            f"backend={backend}, "
            f"out={getattr(args, 'out', 'unknown')}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote long-trace correctness outputs: {args.out}")
    print(json.dumps(result.trace_qa, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
