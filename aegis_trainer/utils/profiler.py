"""
TrainerProfiler — Operation-level timing and performance tracking.

Tracks per-layer and per-operation timing to help identify bottlenecks
in the AEGIS training pipeline. Reports layers/minute, average time per
layer, and detailed per-operation breakdowns.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _TimingEntry:
    """Internal timing record for a single start/end span."""

    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed: float = 0.0
    completed: bool = False


class TrainerProfiler:
    """Hierarchical timing profiler for the AEGIS training pipeline.

    Tracks two levels:
      - Layer-level: How long each full layer takes to process.
      - Operation-level: How long each operation within a layer takes.

    Usage::

        profiler = TrainerProfiler()

        for layer_name in layer_names:
            profiler.start_layer(layer_name)

            profiler.start_operation("load")
            state_dict = layer_io.load(ctx)
            profiler.end_operation("load")

            profiler.start_operation("abliteration")
            state_dict = op.apply(state_dict, ctx)
            profiler.end_operation("abliteration")

            profiler.start_operation("save")
            layer_io.save(state_dict, ctx)
            profiler.end_operation("save")

            profiler.end_layer(layer_name)

        report = profiler.get_report()
    """

    def __init__(self):
        """Initialize the profiler with empty tracking state."""
        self._global_start: float = time.time()

        # Layer-level tracking
        self._layer_timings: Dict[str, _TimingEntry] = {}
        self._layer_order: List[str] = []

        # Operation-level tracking (aggregated across all layers)
        self._op_timings: Dict[str, List[float]] = defaultdict(list)

        # Current in-progress entries
        self._current_layer: Optional[str] = None
        self._current_ops: Dict[str, _TimingEntry] = {}

    def start_layer(self, layer_name: str) -> None:
        """Begin timing a layer.

        Args:
            layer_name: Name of the layer being processed (e.g. "model.layers.0.").
        """
        if self._current_layer is not None:
            logger.warning(
                "start_layer(%s) called while layer %s is still active. "
                "Auto-ending previous layer.",
                layer_name,
                self._current_layer,
            )
            self.end_layer(self._current_layer)

        entry = _TimingEntry(name=layer_name, start_time=time.time())
        self._layer_timings[layer_name] = entry
        self._current_layer = layer_name
        self._current_ops.clear()

        logger.debug("Profiler: started layer %s", layer_name)

    def end_layer(self, layer_name: str) -> float:
        """End timing a layer.

        Args:
            layer_name: Name of the layer to stop timing.

        Returns:
            Elapsed time in seconds for this layer.

        Raises:
            KeyError: If start_layer() was not called for this layer_name.
        """
        if layer_name not in self._layer_timings:
            raise KeyError(
                f"end_layer({layer_name!r}) called but start_layer() was never called."
            )

        entry = self._layer_timings[layer_name]
        entry.end_time = time.time()
        entry.elapsed = entry.end_time - entry.start_time
        entry.completed = True

        # End any still-running operations
        for op_name, op_entry in list(self._current_ops.items()):
            if not op_entry.completed:
                logger.warning(
                    "Operation %s was not ended before end_layer(%s). Auto-ending.",
                    op_name,
                    layer_name,
                )
                self.end_operation(op_name)

        if layer_name not in self._layer_order:
            self._layer_order.append(layer_name)

        self._current_layer = None
        self._current_ops.clear()

        logger.debug(
            "Profiler: ended layer %s (%.2fs)", layer_name, entry.elapsed
        )
        return entry.elapsed

    def start_operation(self, op_name: str) -> None:
        """Begin timing an operation within the current layer.

        Args:
            op_name: Name of the operation (e.g. "load", "abliteration", "save").
        """
        entry = _TimingEntry(name=op_name, start_time=time.time())
        self._current_ops[op_name] = entry
        logger.debug("Profiler: started operation %s", op_name)

    def end_operation(self, op_name: str) -> float:
        """End timing an operation.

        Args:
            op_name: Name of the operation to stop timing.

        Returns:
            Elapsed time in seconds for this operation.

        Raises:
            KeyError: If start_operation() was not called for this op_name.
        """
        if op_name not in self._current_ops:
            raise KeyError(
                f"end_operation({op_name!r}) called but start_operation() was never called."
            )

        entry = self._current_ops[op_name]
        entry.end_time = time.time()
        entry.elapsed = entry.end_time - entry.start_time
        entry.completed = True

        # Record in aggregate stats
        self._op_timings[op_name].append(entry.elapsed)

        logger.debug(
            "Profiler: ended operation %s (%.3fs)", op_name, entry.elapsed
        )
        return entry.elapsed

    @property
    def total_elapsed(self) -> float:
        """Total wall-clock time since profiler was created, in seconds."""
        return time.time() - self._global_start

    @property
    def completed_layers(self) -> int:
        """Number of layers that have been fully timed (start + end)."""
        return sum(1 for e in self._layer_timings.values() if e.completed)

    @property
    def layers_per_minute(self) -> float:
        """Current processing rate in layers per minute.

        Returns:
            Layers per minute, or 0.0 if no layers have been completed.
        """
        elapsed = self.total_elapsed
        if elapsed <= 0 or self.completed_layers == 0:
            return 0.0
        return (self.completed_layers / elapsed) * 60.0

    @property
    def avg_layer_time(self) -> float:
        """Average time per completed layer in seconds.

        Returns:
            Average seconds per layer, or 0.0 if no layers completed.
        """
        completed = [e for e in self._layer_timings.values() if e.completed]
        if not completed:
            return 0.0
        return sum(e.elapsed for e in completed) / len(completed)

    def estimated_remaining(self, total_layers: int) -> float:
        """Estimate remaining time based on current processing rate.

        Args:
            total_layers: Total number of layers to process.

        Returns:
            Estimated remaining seconds, or 0.0 if no data.
        """
        remaining = total_layers - self.completed_layers
        if remaining <= 0 or self.avg_layer_time <= 0:
            return 0.0
        return remaining * self.avg_layer_time

    def get_report(self) -> Dict[str, Any]:
        """Generate a comprehensive profiling report.

        Returns:
            Dictionary with:
              - total_elapsed: Total wall time in seconds.
              - completed_layers: Number of completed layers.
              - layers_per_minute: Processing rate.
              - avg_layer_time: Average time per layer in seconds.
              - per_layer: Dict mapping layer_name -> elapsed seconds.
              - per_operation: Dict mapping op_name -> {
                    count, total, avg, min, max
                } (all in seconds).
        """
        # Per-layer breakdown
        per_layer = {}
        for name in self._layer_order:
            entry = self._layer_timings.get(name)
            if entry and entry.completed:
                per_layer[name] = round(entry.elapsed, 4)

        # Per-operation aggregate stats
        per_operation = {}
        for op_name, times in sorted(self._op_timings.items()):
            if not times:
                continue
            per_operation[op_name] = {
                "count": len(times),
                "total": round(sum(times), 4),
                "avg": round(sum(times) / len(times), 4),
                "min": round(min(times), 4),
                "max": round(max(times), 4),
            }

        return {
            "total_elapsed": round(self.total_elapsed, 2),
            "completed_layers": self.completed_layers,
            "layers_per_minute": round(self.layers_per_minute, 2),
            "avg_layer_time": round(self.avg_layer_time, 4),
            "per_layer": per_layer,
            "per_operation": per_operation,
        }

    def get_summary(self) -> str:
        """Get a human-readable one-line summary of current profiling state.

        Returns:
            Summary string like "12/48 layers (2.5 layers/min, avg 24.0s/layer)".
        """
        return (
            f"{self.completed_layers} layers done "
            f"({self.layers_per_minute:.1f} layers/min, "
            f"avg {self.avg_layer_time:.1f}s/layer, "
            f"elapsed {self.total_elapsed:.0f}s)"
        )
