"""Configuration loading helpers for experiment entry points."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when a configuration file cannot be loaded safely."""


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file as a dictionary.

    The project uses YAML for experiment configs, but every command expects the
    top-level document to be a mapping with string keys. Errors include the path
    being loaded so batch jobs fail with enough context to debug quickly.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"Config file does not exist: {config_path}")
    if not config_path.is_file():
        raise ConfigError(f"Config path is not a file: {config_path}")

    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only without dependency
        raise ConfigError("PyYAML is required to load experiment config files.") from exc

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML config {config_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Could not read config file {config_path}: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Config {config_path} must contain a top-level mapping.")

    config: dict[str, Any] = {}
    for key, value in loaded.items():
        if not isinstance(key, str):
            raise ConfigError(f"Config {config_path} contains a non-string key: {key!r}")
        config[key] = value
    return config
