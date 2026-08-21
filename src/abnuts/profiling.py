"""Optional timing and profiler-marker helpers for benchmark runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Any, TypeVar

import jax

from abnuts.blocking import block_until_ready_tree

T = TypeVar("T")


@dataclass(frozen=True)
class TimingBreakdown:
    """Runtime components captured for an optional profiled benchmark run."""

    enabled: bool = False
    cold_run_seconds: float = 0.0
    cold_compile_seconds: float = 0.0
    warm_iteration_seconds: float = 0.0
    planner_seconds: float = 0.0
    executor_seconds: float = 0.0
    gather_seconds: float = 0.0
    scatter_seconds: float = 0.0
    hvp_seconds: float = 0.0
    total_profiled_seconds: float = 0.0
    profiler_marker_count: int = 0

    def with_outer_timings(
        self,
        *,
        cold_run_seconds: float,
        warm_iteration_seconds: float,
    ) -> TimingBreakdown:
        """Return a copy with cold/warm top-level timing fields filled."""
        cold_compile_seconds = max(0.0, cold_run_seconds - warm_iteration_seconds)
        return replace(
            self,
            enabled=True,
            cold_run_seconds=cold_run_seconds,
            cold_compile_seconds=cold_compile_seconds,
            warm_iteration_seconds=warm_iteration_seconds,
        )

    @property
    def warm_component_seconds(self) -> float:
        """Return measured warm-run components, excluding cold compile overhead."""
        return (
            self.planner_seconds
            + self.executor_seconds
            + self.gather_seconds
            + self.scatter_seconds
            + self.hvp_seconds
        )

    @property
    def unattributed_seconds(self) -> float:
        """Return warm time not assigned to a named component."""
        return max(0.0, self.warm_iteration_seconds - self.warm_component_seconds)

    def as_summary_fields(self, *, elapsed_seconds: float) -> dict[str, float | int | bool]:
        """Return stable CSV fields for benchmark summaries."""
        warm_seconds = self.warm_iteration_seconds if self.enabled else elapsed_seconds
        return {
            "timing_breakdown_enabled": self.enabled,
            "timing_cold_run_seconds": self.cold_run_seconds,
            "timing_cold_compile_seconds": self.cold_compile_seconds,
            "timing_warm_iteration_seconds": warm_seconds,
            "timing_planner_seconds": self.planner_seconds,
            "timing_executor_seconds": self.executor_seconds,
            "timing_gather_seconds": self.gather_seconds,
            "timing_scatter_seconds": self.scatter_seconds,
            "timing_hvp_seconds": self.hvp_seconds,
            "timing_total_profiled_seconds": self.total_profiled_seconds,
            "timing_unattributed_seconds": self.unattributed_seconds if self.enabled else 0.0,
            "timing_profiler_marker_count": self.profiler_marker_count,
        }


class TimingRecorder:
    """Mutable collector for opt-in timing scopes."""

    def __init__(self, *, enabled: bool, enable_profiler_markers: bool = False) -> None:
        self.enabled = enabled
        self.enable_profiler_markers = enable_profiler_markers
        self._seconds = {
            "planner": 0.0,
            "executor": 0.0,
            "gather": 0.0,
            "scatter": 0.0,
            "hvp": 0.0,
        }
        self._profiler_marker_count = 0

    def timed_call(
        self,
        name: str,
        function: Callable[[], T],
        *,
        block_before: Any | None = None,
        block_result: bool = True,
        marker_name: str | None = None,
    ) -> tuple[T, float]:
        """Run ``function`` and record elapsed seconds when timing is enabled."""
        if not self.enabled:
            return function(), 0.0
        if name not in self._seconds:
            raise ValueError(f"unknown timing scope {name!r}")
        if block_before is not None:
            block_until_ready_tree(block_before)

        start = time.perf_counter()
        with self._profiler_marker(marker_name or f"abnuts:{name}"):
            result = function()
            if block_result:
                result = block_until_ready_tree(result)
        elapsed_seconds = time.perf_counter() - start

        self._seconds[name] += elapsed_seconds
        return result, elapsed_seconds

    def snapshot(self) -> TimingBreakdown:
        """Return an immutable snapshot of collected timing components."""
        total_profiled_seconds = sum(self._seconds.values())
        return TimingBreakdown(
            enabled=self.enabled,
            planner_seconds=self._seconds["planner"],
            executor_seconds=self._seconds["executor"],
            gather_seconds=self._seconds["gather"],
            scatter_seconds=self._seconds["scatter"],
            hvp_seconds=self._seconds["hvp"],
            total_profiled_seconds=total_profiled_seconds,
            profiler_marker_count=self._profiler_marker_count,
        )

    def _profiler_marker(self, marker_name: str):
        if not self.enable_profiler_markers:
            return nullcontext()
        annotation = getattr(jax.profiler, "TraceAnnotation", None)
        if annotation is None:
            return nullcontext()
        self._profiler_marker_count += 1
        return annotation(marker_name)
