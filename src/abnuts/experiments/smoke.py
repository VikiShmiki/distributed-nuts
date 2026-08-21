"""Minimal CPU-only smoke command for repository and model scaffolding."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

from abnuts.config import ConfigError, load_yaml_config
from abnuts.io import build_result_manifest, write_manifest
from abnuts.models import available_models, get_model, vectorized_log_prob

MANIFEST_MATCH_KEYS = (
    "schema_version",
    "command",
    "config_sha256",
    "backend",
    "seed",
    "num_chains",
    "dimension",
    "num_warmup",
    "num_draws",
    "placeholder",
)

OPTIONAL_LEGACY_MATCH_KEYS = ("config_hash", "model")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the smoke command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=["cpu"], default=None)
    parser.add_argument("--num-chains", type=int, default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument("--num-warmup", type=int, default=None)
    parser.add_argument("--num-draws", type=int, default=None)
    parser.add_argument(
        "--model",
        choices=available_models(),
        default=None,
        help="Optional benchmark model to validate during smoke.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Apply command-line overrides to the smoke config."""
    merged = dict(config)
    for arg_name, config_name in (
        ("backend", "backend"),
        ("num_chains", "num_chains"),
        ("dimension", "dimension"),
        ("num_warmup", "num_warmup"),
        ("num_draws", "num_draws"),
        ("model", "model"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            merged[config_name] = value
    return merged


def validate_config(config: dict[str, Any]) -> None:
    """Validate the smoke configuration."""
    backend = config.get("backend", "cpu")
    if backend != "cpu":
        raise ValueError(f"Smoke placeholder only supports backend='cpu', got {backend!r}")

    for name in ("seed", "num_chains", "dimension", "num_warmup", "num_draws"):
        value = config.get(name)
        if not isinstance(value, int):
            raise ValueError(f"Config field {name!r} must be an integer, got {value!r}")
        if name != "seed" and value <= 0:
            raise ValueError(f"Config field {name!r} must be positive, got {value!r}")

    model_name = config.get("model")
    if model_name is not None and model_name not in available_models():
        available = ", ".join(available_models())
        raise ValueError(f"Unknown model {model_name!r}. Available models: {available}")


def placeholder_computation(config: dict[str, Any]) -> dict[str, int]:
    """Run a deterministic CPU placeholder so the smoke path does real work."""
    seed = int(config["seed"])
    num_chains = int(config["num_chains"])
    dimension = int(config["dimension"])
    num_iterations = int(config["num_warmup"]) + int(config["num_draws"])

    checksum = 0
    for chain in range(num_chains):
        for dim in range(dimension):
            checksum += (seed + 1) * (chain + 1) * (dim + 1)

    return {
        "checksum": checksum,
        "num_iterations": num_iterations,
        "total_placeholder_updates": checksum * num_iterations,
    }


def model_smoke_check(config: dict[str, Any]) -> dict[str, Any] | None:
    """Run a finite log-probability check for a configured model, if present."""
    model_name = config.get("model")
    if model_name is None:
        return None

    model = get_model(str(model_name), dimension=int(config["dimension"]))
    positions = model.initial_position(
        key=int(config["seed"]),
        num_chains=int(config["num_chains"]),
        config=config,
    )
    log_probs = vectorized_log_prob(model, positions)
    if not all(math.isfinite(value) for value in log_probs):
        raise ValueError(f"Model {model.name!r} produced non-finite initial log probabilities.")

    return {
        "model": model.name,
        "model_metadata": model.metadata.as_dict(),
        "model_check": {
            "num_positions": len(positions),
            "position_dimension": model.dimension,
            "initial_log_prob_min": min(log_probs),
            "initial_log_prob_max": max(log_probs),
            "initial_log_prob_mean": sum(log_probs) / len(log_probs),
        },
    }


def build_manifest(config_path: Path, out_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Build the smoke manifest."""
    model_summary = model_smoke_check(config)
    extra: dict[str, Any] = {
        "backend": config["backend"],
        "seed": config["seed"],
        "num_chains": config["num_chains"],
        "dimension": config["dimension"],
        "num_warmup": config["num_warmup"],
        "num_draws": config["num_draws"],
        "placeholder": placeholder_computation(config),
    }
    if model_summary is not None:
        extra.update(model_summary)

    return build_result_manifest(
        command="python -m abnuts.experiments.smoke",
        config_path=config_path,
        output_dir=out_dir,
        config=config,
        extra=extra,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the smoke command."""
    args = parse_args(argv)
    try:
        config = apply_overrides(load_yaml_config(args.config), args)
        validate_config(config)
        manifest = build_manifest(args.config, args.out, config)
        manifest_path, wrote_manifest = write_manifest(
            args.out,
            manifest,
            overwrite=args.overwrite,
            comparable_keys=MANIFEST_MATCH_KEYS,
            optional_existing_keys=OPTIONAL_LEGACY_MATCH_KEYS,
        )
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Smoke command failed: {exc}", file=sys.stderr)
        return 2

    if wrote_manifest:
        print(f"Wrote smoke manifest: {manifest_path}")
    else:
        print(f"Preserved existing smoke manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
