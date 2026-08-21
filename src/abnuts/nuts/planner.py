"""Host-side bucket planning for fixed-shape NUTS scheduling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
from jax import Array


class BucketPlan(NamedTuple):
    """A padded bucket schedule over real chain indices.

    ``idx`` and ``mask`` are rectangular so the plan has a stable shape.  For
    mixed canonical sizes, ``bucket_sizes`` records the executor width intended
    for each row; false mask entries beyond that width are representation padding
    and are not included in ``bucket_padding_count``.
    """

    idx: Array
    mask: Array
    bucket_sizes: Array
    occupancy: Array
    num_buckets: int
    predicted_work: Array
    sorted_indices: Array
    padding_count: Array
    padding_ratio: Array
    bucket_padding_count: Array
    bucket_min_predicted_work: Array
    bucket_max_predicted_work: Array


def make_bucket_plan(
    predicted_work: Array,
    *,
    canonical_bucket_sizes: int | Sequence[int],
) -> BucketPlan:
    """Sort chains by predicted work and pack them into padded canonical buckets.

    The planner is intentionally host-side: it consumes predicted per-chain work
    scores, performs a stable sort, chooses canonical bucket capacities with
    minimal padding and then minimal bucket count, and emits real-chain indices
    plus masks.  Padded lanes repeat the last valid chain index in their bucket.
    """
    work = _validate_predicted_work(predicted_work)
    canonical_sizes = _normalize_bucket_sizes(canonical_bucket_sizes)
    bucket_sizes = _choose_bucket_sizes(work.shape[0], canonical_sizes)

    sorted_indices = np.argsort(work, kind="stable").astype(np.int32)
    max_bucket_size = max(canonical_sizes)
    num_buckets = len(bucket_sizes)
    idx = np.empty((num_buckets, max_bucket_size), dtype=np.int32)
    mask = np.zeros((num_buckets, max_bucket_size), dtype=np.bool_)
    occupancy = np.empty((num_buckets,), dtype=np.int32)
    bucket_min = np.empty((num_buckets,), dtype=work.dtype)
    bucket_max = np.empty((num_buckets,), dtype=work.dtype)

    start = 0
    for bucket_number, bucket_size in enumerate(bucket_sizes):
        stop = min(start + bucket_size, sorted_indices.shape[0])
        real_indices = sorted_indices[start:stop]
        if real_indices.size == 0:
            raise ValueError("internal planner error: empty bucket")

        fill_index = real_indices[-1]
        idx[bucket_number, :] = fill_index
        idx[bucket_number, : real_indices.size] = real_indices
        mask[bucket_number, : real_indices.size] = True
        occupancy[bucket_number] = real_indices.size

        bucket_work = work[real_indices]
        bucket_min[bucket_number] = np.min(bucket_work)
        bucket_max[bucket_number] = np.max(bucket_work)
        start = stop

    if start != sorted_indices.shape[0]:
        raise ValueError("internal planner error: not all chains were assigned")

    bucket_sizes_array = np.asarray(bucket_sizes, dtype=np.int32)
    bucket_padding = bucket_sizes_array - occupancy
    padding_count = np.asarray(np.sum(bucket_padding), dtype=np.int32)
    total_bucket_lanes = np.asarray(np.sum(bucket_sizes_array), dtype=work.dtype)
    padding_ratio = np.asarray(padding_count, dtype=work.dtype) / total_bucket_lanes

    return BucketPlan(
        idx=jnp.asarray(idx),
        mask=jnp.asarray(mask),
        bucket_sizes=jnp.asarray(bucket_sizes_array),
        occupancy=jnp.asarray(occupancy),
        num_buckets=num_buckets,
        predicted_work=jnp.asarray(work),
        sorted_indices=jnp.asarray(sorted_indices),
        padding_count=jnp.asarray(padding_count),
        padding_ratio=jnp.asarray(padding_ratio),
        bucket_padding_count=jnp.asarray(bucket_padding),
        bucket_min_predicted_work=jnp.asarray(bucket_min),
        bucket_max_predicted_work=jnp.asarray(bucket_max),
    )


def _validate_predicted_work(predicted_work: Array) -> np.ndarray:
    work = np.asarray(predicted_work)
    if work.ndim != 1:
        raise ValueError(f"predicted_work must be one-dimensional; got shape {work.shape}")
    if work.shape[0] == 0:
        raise ValueError("predicted_work must contain at least one chain")
    if not np.all(np.isfinite(work)):
        raise ValueError("predicted_work must contain only finite values")
    return work


def _normalize_bucket_sizes(canonical_bucket_sizes: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(canonical_bucket_sizes, int):
        raw_sizes = (canonical_bucket_sizes,)
    else:
        raw_sizes = tuple(canonical_bucket_sizes)

    if len(raw_sizes) == 0:
        raise ValueError("canonical_bucket_sizes must contain at least one size")

    sizes = tuple(sorted({int(size) for size in raw_sizes}))
    if any(size <= 0 for size in sizes):
        raise ValueError(f"canonical_bucket_sizes must be positive; got {raw_sizes!r}")
    return sizes


def _choose_bucket_sizes(num_chains: int, canonical_sizes: tuple[int, ...]) -> tuple[int, ...]:
    max_bucket_size = canonical_sizes[-1]
    search_limit = num_chains + max_bucket_size - 1
    sequences: list[tuple[int, ...] | None] = [None] * (search_limit + 1)
    sequences[0] = ()

    for capacity in range(1, search_limit + 1):
        candidates: list[tuple[int, ...]] = []
        for bucket_size in canonical_sizes:
            previous_capacity = capacity - bucket_size
            if previous_capacity < 0:
                continue
            previous = sequences[previous_capacity]
            if previous is not None:
                candidates.append((*previous, bucket_size))
        if candidates:
            sequences[capacity] = min(
                candidates,
                key=lambda sequence: (len(sequence), tuple(-size for size in sequence)),
            )

    valid_sequences = [
        (capacity - num_chains, len(sequence), sequence)
        for capacity, sequence in enumerate(sequences[num_chains:], start=num_chains)
        if sequence is not None
    ]
    if not valid_sequences:
        raise ValueError("internal planner error: could not cover chain count with buckets")

    _, _, best_sequence = min(valid_sequences, key=lambda item: (item[0], item[1]))
    return tuple(sorted(best_sequence, reverse=True))
