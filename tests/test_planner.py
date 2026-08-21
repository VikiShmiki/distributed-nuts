from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from abnuts.nuts.planner import make_bucket_plan


def _real_indices(plan) -> list[int]:
    idx = np.asarray(plan.idx)
    mask = np.asarray(plan.mask)
    return [int(index) for index in idx[mask]]


def test_planner_handles_one_chain_with_padding() -> None:
    plan = make_bucket_plan(
        jnp.asarray([2.0], dtype=jnp.float32),
        canonical_bucket_sizes=4,
    )

    assert plan.num_buckets == 1
    assert plan.idx.shape == (1, 4)
    assert plan.mask.shape == (1, 4)
    assert np.array_equal(np.asarray(plan.idx), np.asarray([[0, 0, 0, 0]]))
    assert np.array_equal(np.asarray(plan.mask), np.asarray([[True, False, False, False]]))
    assert np.array_equal(np.asarray(plan.bucket_sizes), np.asarray([4], dtype=np.int32))
    assert np.array_equal(np.asarray(plan.occupancy), np.asarray([1], dtype=np.int32))
    assert int(plan.padding_count) == 3
    assert float(plan.padding_ratio) == pytest.approx(0.75)
    assert _real_indices(plan) == [0]


def test_planner_stably_sorts_and_packs_exact_multiple() -> None:
    plan = make_bucket_plan(
        jnp.asarray([2.0, 1.0, 1.0, 3.0], dtype=jnp.float32),
        canonical_bucket_sizes=2,
    )

    assert plan.num_buckets == 2
    assert np.array_equal(np.asarray(plan.sorted_indices), np.asarray([1, 2, 0, 3]))
    assert np.array_equal(np.asarray(plan.idx), np.asarray([[1, 2], [0, 3]]))
    assert np.array_equal(np.asarray(plan.mask), np.ones((2, 2), dtype=np.bool_))
    assert np.array_equal(np.asarray(plan.bucket_padding_count), np.asarray([0, 0]))
    assert int(plan.padding_count) == 0
    assert float(plan.padding_ratio) == pytest.approx(0.0)
    assert sorted(_real_indices(plan)) == [0, 1, 2, 3]


def test_planner_pads_non_multiple_by_repeating_last_valid_index() -> None:
    plan = make_bucket_plan(
        jnp.asarray([4.0, 1.0, 3.0, 2.0, 5.0], dtype=jnp.float32),
        canonical_bucket_sizes=2,
    )

    assert plan.num_buckets == 3
    assert np.array_equal(np.asarray(plan.bucket_sizes), np.asarray([2, 2, 2]))
    assert np.array_equal(np.asarray(plan.occupancy), np.asarray([2, 2, 1]))
    assert np.array_equal(np.asarray(plan.idx), np.asarray([[1, 3], [2, 0], [4, 4]]))
    assert np.array_equal(
        np.asarray(plan.mask),
        np.asarray([[True, True], [True, True], [True, False]]),
    )
    assert np.array_equal(np.asarray(plan.bucket_padding_count), np.asarray([0, 0, 1]))
    assert int(plan.padding_count) == 1
    assert float(plan.padding_ratio) == pytest.approx(1.0 / 6.0)
    assert sorted(_real_indices(plan)) == [0, 1, 2, 3, 4]


def test_planner_supports_multiple_canonical_sizes() -> None:
    plan = make_bucket_plan(
        jnp.asarray([5.0, 1.0, 6.0, 2.0, 7.0, 3.0], dtype=jnp.float32),
        canonical_bucket_sizes=[2, 4],
    )

    assert plan.num_buckets == 2
    assert plan.idx.shape == (2, 4)
    assert np.array_equal(np.asarray(plan.bucket_sizes), np.asarray([4, 2]))
    assert np.array_equal(np.asarray(plan.occupancy), np.asarray([4, 2]))
    assert np.array_equal(np.asarray(plan.idx), np.asarray([[1, 3, 5, 0], [2, 4, 4, 4]]))
    assert np.array_equal(
        np.asarray(plan.mask),
        np.asarray([[True, True, True, True], [True, True, False, False]]),
    )
    assert np.array_equal(np.asarray(plan.bucket_padding_count), np.asarray([0, 0]))
    assert int(plan.padding_count) == 0
    assert float(plan.padding_ratio) == pytest.approx(0.0)
    assert np.array_equal(np.asarray(plan.bucket_min_predicted_work), np.asarray([1.0, 6.0]))
    assert np.array_equal(np.asarray(plan.bucket_max_predicted_work), np.asarray([5.0, 7.0]))
    assert sorted(_real_indices(plan)) == [0, 1, 2, 3, 4, 5]


def test_planner_validation_errors_are_informative() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        make_bucket_plan(jnp.ones((2, 2)), canonical_bucket_sizes=2)
    with pytest.raises(ValueError, match="at least one chain"):
        make_bucket_plan(jnp.asarray([], dtype=jnp.float32), canonical_bucket_sizes=2)
    with pytest.raises(ValueError, match="finite"):
        make_bucket_plan(jnp.asarray([1.0, jnp.inf]), canonical_bucket_sizes=2)
    with pytest.raises(ValueError, match="at least one size"):
        make_bucket_plan(jnp.ones((2,)), canonical_bucket_sizes=[])
    with pytest.raises(ValueError, match="positive"):
        make_bucket_plan(jnp.ones((2,)), canonical_bucket_sizes=[2, 0])
