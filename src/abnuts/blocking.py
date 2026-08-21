"""Helpers for blocking asynchronous array work before timing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def block_until_ready_tree(value: Any) -> Any:
    """Block all JAX leaves in a pytree, with a pure-Python fallback.

    JAX dispatch is asynchronous, so timing code must force returned arrays to
    finish before reading the clock. When JAX is not installed, this still walks
    ordinary Python containers and calls ``block_until_ready`` on any compatible
    leaf object.
    """
    try:
        from jax import tree_util
    except ModuleNotFoundError:
        return _block_until_ready_tree_without_jax(value)
    return tree_util.tree_map(_block_leaf, value)


def _block_until_ready_tree_without_jax(value: Any) -> Any:
    if isinstance(value, Mapping):
        return type(value)(
            (key, _block_until_ready_tree_without_jax(item)) for key, item in value.items()
        )
    if isinstance(value, tuple):
        return type(value)(_block_until_ready_tree_without_jax(item) for item in value)
    if isinstance(value, list):
        return [_block_until_ready_tree_without_jax(item) for item in value]
    return _block_leaf(value)


def _block_leaf(value: Any) -> Any:
    block = getattr(value, "block_until_ready", None)
    if callable(block):
        return block()
    return value
