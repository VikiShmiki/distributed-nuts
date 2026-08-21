"""Run a small simulation-based calibration validation skeleton."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from abnuts.blocking import block_until_ready_tree
from abnuts.config import ConfigError, load_yaml_config
from abnuts.io import build_result_manifest, write_manifest
from abnuts.models.base import ModelMetadata
from abnuts.nuts.bucketed import BucketedRunResult, run_bucketed
from abnuts.nuts.monolithic import MonolithicRunResult, run_monolithic
from abnuts.nuts.predictors import PREDICTOR_MODES

LOG_TWO_PI = math.log(2.0 * math.pi)
MANIFEST_MATCH_KEYS = (
    "schema_version",
    "command",
    "config_sha256",
    "config_hash",
    "profile",
    "benchmark",
    "model",
)
METHODS = frozenset({"monolithic", "bucketed"})


@dataclass(frozen=True)
class NormalLocationSBCModel:
    """Conjugate normal-location SBC target with one unknown scalar parameter."""

    dimension: int
    prior_scale: float
    observation_scale: float
    num_observations: int
    initial_jitter_scale: float = 0.05
    name: str = "normal_location"

    def __post_init__(self) -> None:
        """Validate normal-location dimensions and scales."""
        if self.dimension != 1:
            raise ValueError(f"normal_location SBC requires dimension=1, got {self.dimension!r}")
        if self.prior_scale <= 0.0:
            raise ValueError(f"prior_scale must be positive, got {self.prior_scale!r}")
        if self.observation_scale <= 0.0:
            raise ValueError(
                f"observation_scale must be positive, got {self.observation_scale!r}"
            )
        if self.num_observations <= 0:
            raise ValueError(
                f"num_observations must be positive, got {self.num_observations!r}"
            )
        if self.initial_jitter_scale <= 0.0:
            raise ValueError(
                f"initial_jitter_scale must be positive, got {self.initial_jitter_scale!r}"
            )

    @property
    def metadata(self) -> ModelMetadata:
        """Return serializable metadata for SBC manifests."""
        return ModelMetadata(
            name=self.name,
            dimension=self.dimension,
            event_shape=(self.dimension,),
            description=(
                "Normal-location SBC model with theta ~ Normal(0, prior_scale) "
                "and y_i | theta ~ Normal(theta, observation_scale)."
            ),
            extra={
                "model_family": "sbc",
                "num_observations": self.num_observations,
                "prior_scale": self.prior_scale,
                "observation_scale": self.observation_scale,
            },
        )

    def initial_position(
        self,
        key: int,
        num_chains: int,
        config: dict[str, Any] | None = None,
    ) -> list[list[float]]:
        """Generate deterministic scalar initial positions near zero."""
        if num_chains <= 0:
            raise ValueError(f"num_chains must be positive, got {num_chains!r}")
        jitter_scale = float((config or {}).get("initial_jitter_scale", self.initial_jitter_scale))
        if jitter_scale <= 0.0:
            raise ValueError(f"initial_jitter_scale must be positive, got {jitter_scale!r}")
        rng = random.Random(int(key))
        return [[rng.gauss(0.0, jitter_scale)] for _ in range(num_chains)]

    def log_prob(self, position: Sequence[float] | Any, data: Any | None = None) -> Any:
        """Evaluate the scalar posterior log density for one simulated dataset."""
        if data is None:
            raise ValueError("normal_location SBC log_prob requires simulated data")
        if hasattr(position, "shape"):
            position_array = jnp.asarray(position)
            if position_array.shape != (self.dimension,):
                raise ValueError(
                    f"Expected position with shape ({self.dimension},), "
                    f"got {position_array.shape}"
                )
            theta = position_array[0]
            observations = jnp.asarray(data["y"], dtype=position_array.dtype)
            log_prob = _jax_normal_log_prob(theta, 0.0, self.prior_scale)
            log_prob += jnp.sum(
                _jax_normal_log_prob(observations, theta, self.observation_scale)
            )
            return log_prob

        if len(position) != self.dimension:
            raise ValueError(
                f"Expected position with dimension {self.dimension}, got {len(position)}"
            )
        theta = float(position[0])
        observations = _python_observations(data)
        log_prob = _normal_log_prob(theta, 0.0, self.prior_scale)
        log_prob += sum(
            _normal_log_prob(observed, theta, self.observation_scale)
            for observed in observations
        )
        return log_prob


class SBCConfig(NamedTuple):
    """Resolved SBC configuration for one profile."""

    benchmark: str
    model: str
    backend: str
    dtype: str
    seed: int
    methods: tuple[str, ...]
    num_replicates: int
    num_chains: int
    num_warmup: int
    num_draws: int
    prior_scale: float
    observation_scale: float
    num_observations: int
    step_size: float
    max_tree_depth: int
    bucket_size: int
    predictor: str
    predictor_beta: float
    initial_jitter_scale: float
    smoke_only: bool


class SBCRankRow(NamedTuple):
    """One SBC rank result for one replicate and method."""

    replicate: int
    method: str
    true_parameter: float
    rank: int
    num_posterior_samples: int
    posterior_mean: float
    posterior_sd: float
    divergence_count: int
    max_tree_depth_hit_count: int
    data_mean: float


class SBCResult(NamedTuple):
    """Raw SBC rows and metadata."""

    ranks: tuple[SBCRankRow, ...]
    rank_histogram_rows: tuple[dict[str, Any], ...]
    model_metadata: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the SBC runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--profile", default="tiny")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def resolve_config(raw_config: dict[str, Any], *, profile: str) -> SBCConfig:
    """Resolve top-level defaults plus one named profile into a typed SBC config."""
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
    config = SBCConfig(
        benchmark=str(merged.get("benchmark", "sbc")),
        model=str(merged.get("model", "normal_location")),
        backend=str(merged.get("backend", "cpu")),
        dtype=str(merged.get("dtype", "float32")),
        seed=_as_int(merged.get("seed", 0), "seed"),
        methods=_as_methods(merged.get("methods", ["monolithic", "bucketed"])),
        num_replicates=_as_positive_int(merged.get("num_replicates", 2), "num_replicates"),
        num_chains=_as_positive_int(merged.get("num_chains", 2), "num_chains"),
        num_warmup=_as_nonnegative_int(merged.get("num_warmup", 1), "num_warmup"),
        num_draws=_as_positive_int(merged.get("num_draws", 2), "num_draws"),
        prior_scale=_as_positive_float(merged.get("prior_scale", 1.0), "prior_scale"),
        observation_scale=_as_positive_float(
            merged.get("observation_scale", 1.0),
            "observation_scale",
        ),
        num_observations=_as_positive_int(
            merged.get("num_observations", 4),
            "num_observations",
        ),
        step_size=_as_positive_float(merged.get("step_size", 0.04), "step_size"),
        max_tree_depth=_as_positive_int(merged.get("max_tree_depth", 2), "max_tree_depth"),
        bucket_size=_as_positive_int(merged.get("bucket_size", 2), "bucket_size"),
        predictor=str(merged.get("predictor", "history")),
        predictor_beta=_as_probability(merged.get("predictor_beta", 0.9), "predictor_beta"),
        initial_jitter_scale=_as_positive_float(
            merged.get("initial_jitter_scale", 0.05),
            "initial_jitter_scale",
        ),
        smoke_only=_as_bool(merged.get("smoke_only", profile == "tiny")),
    )
    validate_config(config)
    return config


def validate_config(config: SBCConfig) -> None:
    """Validate resolved SBC settings."""
    if config.benchmark != "sbc":
        raise ValueError(f"expected benchmark='sbc', got {config.benchmark!r}")
    if config.model != "normal_location":
        raise ValueError(
            "run_sbc currently supports model='normal_location'; "
            f"got {config.model!r}"
        )
    if config.backend != "cpu":
        raise ValueError(f"run_sbc currently supports backend='cpu', got {config.backend!r}")
    if config.dtype != "float32":
        raise ValueError(f"run_sbc currently supports dtype='float32', got {config.dtype!r}")
    if config.predictor not in PREDICTOR_MODES:
        valid = ", ".join(sorted(PREDICTOR_MODES))
        raise ValueError(f"unsupported predictor {config.predictor!r}; expected one of {valid}")


def run_sbc(config: SBCConfig) -> SBCResult:
    """Run a tiny SBC loop and return rank rows plus histogram rows."""
    model = NormalLocationSBCModel(
        dimension=1,
        prior_scale=config.prior_scale,
        observation_scale=config.observation_scale,
        num_observations=config.num_observations,
        initial_jitter_scale=config.initial_jitter_scale,
    )
    rank_rows: list[SBCRankRow] = []
    for replicate in range(config.num_replicates):
        replicate_seed = config.seed + 10_003 * replicate
        data = _simulate_normal_location_data(model, seed=replicate_seed)
        initial_positions = jnp.asarray(
            model.initial_position(
                key=replicate_seed + 17,
                num_chains=config.num_chains,
                config={"initial_jitter_scale": config.initial_jitter_scale},
            ),
            dtype=jnp.float32,
        )
        initial_positions = block_until_ready_tree(initial_positions)
        transition_key = jr.PRNGKey(replicate_seed + 101)
        method_results = _run_methods(
            config,
            model,
            initial_positions,
            transition_key,
            data,
        )
        for method, result in method_results.items():
            rank_rows.append(
                _rank_result(
                    replicate=replicate,
                    method=method,
                    true_parameter=data["theta"],
                    data_mean=float(np.mean(data["y"])),
                    num_warmup=config.num_warmup,
                    result=result,
                )
            )

    histogram_rows = _rank_histogram_rows(
        rank_rows,
        methods=config.methods,
        profile_smoke_only=config.smoke_only,
    )
    return SBCResult(
        ranks=tuple(rank_rows),
        rank_histogram_rows=tuple(histogram_rows),
        model_metadata=model.metadata.as_dict(),
    )


def write_outputs(
    out_dir: Path,
    *,
    config_path: Path,
    profile: str,
    config: SBCConfig,
    result: SBCResult,
    overwrite: bool,
) -> None:
    """Write SBC rank data, rank histogram data, and manifest."""
    resolved_config = _config_to_json(config, profile=profile)
    manifest = build_result_manifest(
        command="python -m abnuts.experiments.run_sbc",
        config_path=config_path,
        output_dir=out_dir,
        config=resolved_config,
        extra={
            "profile": profile,
            "benchmark": config.benchmark,
            "model": config.model,
            "model_metadata": result.model_metadata,
            "smoke_only": config.smoke_only,
            "evidence_note": _evidence_note(config),
            "outputs": {
                "manifest_json": "manifest.json",
                "ranks_csv": "ranks.csv",
                "rank_histogram_csv": "rank_histogram.csv",
            },
        },
    )
    write_manifest(
        out_dir,
        manifest,
        overwrite=overwrite,
        comparable_keys=MANIFEST_MATCH_KEYS,
    )
    _write_text_preserving(out_dir / "ranks.csv", _ranks_csv(result.ranks), overwrite=overwrite)
    _write_text_preserving(
        out_dir / "rank_histogram.csv",
        _dict_rows_csv(result.rank_histogram_rows),
        overwrite=overwrite,
    )


def _run_methods(
    config: SBCConfig,
    model: NormalLocationSBCModel,
    initial_positions: Any,
    rng_key: Any,
    data: dict[str, Any],
) -> dict[str, MonolithicRunResult | BucketedRunResult]:
    num_steps = config.num_warmup + config.num_draws
    results: dict[str, MonolithicRunResult | BucketedRunResult] = {}
    if "monolithic" in config.methods:
        results["monolithic"] = run_monolithic(
            model,
            initial_positions,
            rng_key,
            num_steps=num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            dtype=jnp.float32,
            data=data,
        )
    if "bucketed" in config.methods:
        results["bucketed"] = run_bucketed(
            model,
            initial_positions,
            rng_key,
            num_steps=num_steps,
            step_size=config.step_size,
            max_tree_depth=config.max_tree_depth,
            bucket_size=config.bucket_size,
            predictor=config.predictor,
            predictor_beta=config.predictor_beta,
            dtype=jnp.float32,
            data=data,
        )
    return block_until_ready_tree(results)


def _rank_result(
    *,
    replicate: int,
    method: str,
    true_parameter: float,
    data_mean: float,
    num_warmup: int,
    result: MonolithicRunResult | BucketedRunResult,
) -> SBCRankRow:
    samples = np.asarray(result.trace_positions)[num_warmup:, :, 0].reshape(-1)
    if samples.size == 0:
        raise ValueError("SBC rank computation needs at least one post-warmup draw")
    transition_info = result.transition_info
    return SBCRankRow(
        replicate=replicate,
        method=method,
        true_parameter=float(true_parameter),
        rank=int(np.sum(samples < true_parameter)),
        num_posterior_samples=int(samples.size),
        posterior_mean=float(np.mean(samples)),
        posterior_sd=float(np.std(samples, ddof=1)) if samples.size > 1 else 0.0,
        divergence_count=int(np.sum(np.asarray(transition_info.divergence_flag))),
        max_tree_depth_hit_count=int(np.sum(np.asarray(transition_info.max_tree_depth_hit))),
        data_mean=data_mean,
    )


def _simulate_normal_location_data(
    model: NormalLocationSBCModel,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(int(seed))
    theta = rng.gauss(0.0, model.prior_scale)
    observations = [
        rng.gauss(theta, model.observation_scale) for _ in range(model.num_observations)
    ]
    return {"theta": theta, "y": observations}


def _rank_histogram_rows(
    rows: list[SBCRankRow],
    *,
    methods: tuple[str, ...],
    profile_smoke_only: bool,
) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("cannot build an SBC rank histogram from no rows")
    sample_counts = {row.num_posterior_samples for row in rows}
    if len(sample_counts) != 1:
        raise ValueError("all SBC rows must have the same number of posterior samples")
    num_samples = sample_counts.pop()
    histogram_rows: list[dict[str, Any]] = []
    for method in methods:
        method_rows = [row for row in rows if row.method == method]
        for rank in range(num_samples + 1):
            histogram_rows.append(
                {
                    "method": method,
                    "rank": rank,
                    "count": sum(1 for row in method_rows if row.rank == rank),
                    "num_replicates": len(method_rows),
                    "num_posterior_samples": num_samples,
                    "is_smoke_only": profile_smoke_only,
                    "evidence_note": (
                        "smoke-only; not paper evidence"
                        if profile_smoke_only
                        else "validation run"
                    ),
                }
            )
    return histogram_rows


def _ranks_csv(rows: tuple[SBCRankRow, ...]) -> str:
    return _dict_rows_csv([row._asdict() for row in rows])


def _dict_rows_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(row))
    return buffer.getvalue()


def _config_to_json(config: SBCConfig, *, profile: str) -> dict[str, Any]:
    return {
        "profile": profile,
        "benchmark": config.benchmark,
        "model": config.model,
        "backend": config.backend,
        "dtype": config.dtype,
        "seed": config.seed,
        "methods": list(config.methods),
        "num_replicates": config.num_replicates,
        "num_chains": config.num_chains,
        "num_warmup": config.num_warmup,
        "num_draws": config.num_draws,
        "prior_scale": config.prior_scale,
        "observation_scale": config.observation_scale,
        "num_observations": config.num_observations,
        "step_size": config.step_size,
        "max_tree_depth": config.max_tree_depth,
        "bucket_size": config.bucket_size,
        "predictor": config.predictor,
        "predictor_beta": config.predictor_beta,
        "initial_jitter_scale": config.initial_jitter_scale,
        "smoke_only": config.smoke_only,
    }


def _evidence_note(config: SBCConfig) -> str:
    if config.smoke_only:
        return "Tiny SBC profile is smoke-only and not paper evidence."
    return "SBC validation output."


def _normal_log_prob(value: float, loc: float, scale: float) -> float:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * math.log(scale) + LOG_TWO_PI)


def _jax_normal_log_prob(value: Any, loc: Any, scale: Any) -> Any:
    return -0.5 * (((value - loc) / scale) ** 2 + 2.0 * jnp.log(scale) + LOG_TWO_PI)


def _python_observations(data: Any) -> tuple[float, ...]:
    try:
        observations = tuple(float(value) for value in data["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("normal_location SBC data must contain numeric 'y' values") from exc
    if not observations:
        raise ValueError("normal_location SBC data must contain at least one observation")
    return observations


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


def _as_nonnegative_int(value: Any, name: str) -> int:
    parsed = _as_int(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be nonnegative, got {parsed!r}")
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


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_methods(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        methods = tuple(part.strip() for part in value.split(",") if part.strip())
    elif isinstance(value, Sequence):
        methods = tuple(str(part) for part in value)
    else:
        raise ValueError("methods must be a sequence or comma-separated string")
    if not methods:
        raise ValueError("methods must contain at least one method")
    invalid = sorted(set(methods) - METHODS)
    if invalid:
        valid = ", ".join(sorted(METHODS))
        raise ValueError(f"unsupported SBC method(s) {invalid}; expected methods in {valid}")
    return methods


def _write_text_preserving(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return
        raise FileExistsError(
            f"Output file already exists with different content: {path}. "
            "Pass --overwrite to replace it intentionally."
        )
    path.write_text(content, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run the SBC command."""
    args = parse_args(argv)
    backend = "unknown"
    try:
        raw_config = load_yaml_config(args.config)
        config = resolve_config(raw_config, profile=args.profile)
        backend = config.backend
        jax.config.update("jax_platform_name", config.backend)
        result = run_sbc(config)
        write_outputs(
            args.out,
            config_path=args.config,
            profile=args.profile,
            config=config,
            result=result,
            overwrite=args.overwrite,
        )
    except (ConfigError, OSError, ValueError) as exc:
        print(
            "SBC run failed "
            f"(config={getattr(args, 'config', 'unknown')}, "
            f"profile={getattr(args, 'profile', 'unknown')}, "
            f"backend={backend}, "
            f"out={getattr(args, 'out', 'unknown')}): {exc}",
            file=sys.stderr,
        )
        return 2

    print(f"Wrote SBC outputs: {args.out}")
    print(json.dumps({"smoke_only": config.smoke_only, "rank_rows": len(result.ranks)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
