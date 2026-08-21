from __future__ import annotations

import json
from pathlib import Path

import pytest

from abnuts.blocking import block_until_ready_tree
from abnuts.config import ConfigError, load_yaml_config
from abnuts.io import build_result_manifest, write_manifest


def test_load_yaml_config_returns_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "backend: cpu\nnum_chains: 8\nnested:\n  enabled: true\n",
        encoding="utf-8",
    )

    config = load_yaml_config(config_path)

    assert config["backend"] == "cpu"
    assert config["num_chains"] == 8
    assert config["nested"] == {"enabled": True}


def test_load_yaml_config_rejects_non_mapping(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="top-level mapping"):
        load_yaml_config(config_path)


def test_build_and_write_manifest_protects_outputs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("backend: cpu\nnum_chains: 8\n", encoding="utf-8")
    config = load_yaml_config(config_path)

    manifest = build_result_manifest(
        command="python -m abnuts.experiments.smoke",
        config_path=config_path,
        output_dir=tmp_path / "out",
        config=config,
        extra={"backend": "cpu", "num_chains": 8},
    )

    manifest_path, wrote_manifest = write_manifest(
        tmp_path / "out",
        manifest,
        comparable_keys=("command", "config_sha256", "config_hash", "backend", "num_chains"),
    )

    assert wrote_manifest is True
    written = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert written["schema_version"] == 1
    assert written["command"] == "python -m abnuts.experiments.smoke"
    assert written["config_sha256"]
    assert written["config_hash"]
    assert written["created_at_utc"]
    assert written["python_version"]
    assert written["platform"]

    preserved_path, wrote_again = write_manifest(
        tmp_path / "out",
        manifest,
        comparable_keys=("command", "config_sha256", "config_hash", "backend", "num_chains"),
    )
    assert preserved_path == manifest_path
    assert wrote_again is False

    changed = dict(manifest)
    changed["num_chains"] = 16
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_manifest(
            tmp_path / "out",
            changed,
            comparable_keys=("command", "config_sha256", "config_hash", "backend", "num_chains"),
        )


def test_block_until_ready_tree_fallback_blocks_leaves() -> None:
    class BlockingLeaf:
        def __init__(self) -> None:
            self.blocked = False

        def block_until_ready(self) -> "BlockingLeaf":
            self.blocked = True
            return self

    leaf = BlockingLeaf()
    tree = {"leaf": [leaf], "plain": 1}

    result = block_until_ready_tree(tree)

    assert result["leaf"][0] is leaf
    assert leaf.blocked is True
