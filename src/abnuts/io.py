"""Result schema and JSON writing helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_MANIFEST_MATCH_KEYS = ("schema_version", "command", "config_sha256", "config_hash")


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest for a file."""
    digest = hashlib.sha256()
    digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    """Hash a JSON-serializable value using stable key ordering."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_result_manifest(
    *,
    command: str,
    config_path: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
    status: str = "ok",
    schema_version: int = 1,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest with stable metadata common to all result writers."""
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "command": command,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_hash": stable_json_hash(config),
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_manifest(
    out_dir: str | Path,
    manifest: dict[str, Any],
    *,
    overwrite: bool = False,
    comparable_keys: tuple[str, ...] = DEFAULT_MANIFEST_MATCH_KEYS,
    optional_existing_keys: tuple[str, ...] = (),
) -> tuple[Path, bool]:
    """Write ``manifest.json`` without replacing raw outputs by default.

    Returns the manifest path and a boolean indicating whether a new file was
    written. If an equivalent manifest already exists, the file is preserved.
    """
    output_dir = Path(out_dir)
    manifest_path = output_dir / "manifest.json"

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        if manifest_path.exists() and existing_manifest_matches(
            manifest_path,
            manifest,
            comparable_keys=comparable_keys,
            optional_existing_keys=optional_existing_keys,
        ):
            return manifest_path, False
        raise FileExistsError(
            f"Output directory already contains files: {output_dir}. "
            "Pass --overwrite to replace the manifest intentionally."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}. "
            "Pass --overwrite to replace it intentionally."
        )

    manifest_json = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    manifest_path.write_text(manifest_json, encoding="utf-8")
    return manifest_path, True


def existing_manifest_matches(
    manifest_path: str | Path,
    manifest: dict[str, Any],
    *,
    comparable_keys: tuple[str, ...],
    optional_existing_keys: tuple[str, ...] = (),
) -> bool:
    """Return whether an existing manifest describes the same stable inputs."""
    try:
        existing = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    for key in comparable_keys:
        if existing.get(key) != manifest.get(key):
            return False
    for key in optional_existing_keys:
        if key in existing and existing.get(key) != manifest.get(key):
            return False
    return True
