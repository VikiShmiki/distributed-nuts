from __future__ import annotations

from abnuts.experiments.run_benchmark import _time_runner


def test_unprofiled_timing_primes_once_then_retains_all_warm_repeats() -> None:
    calls = 0

    def runner() -> int:
        nonlocal calls
        calls += 1
        return calls

    timed = _time_runner(
        runner,
        enable_timing_breakdown=False,
        timing_from_result=None,
        warm_repeats=5,
    )

    assert calls == 6
    assert timed.result == 6
    assert timed.cold_prime_seconds >= 0.0
    assert len(timed.repeat_seconds) == 5
    assert timed.elapsed_seconds == sorted(timed.repeat_seconds)[2]
