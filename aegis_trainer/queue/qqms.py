"""
QQMS — Quantized Queue Memory Streaming.

The core queue execution engine for AEGIS AI Trainer. Manages resource-aware
throttling, retry logic, and dead letter queue routing for layer-by-layer
weight modification operations.

Mirrors AirLLM's forward() pattern: load one layer, process it, free memory,
never hold more than one layer in RAM at a time.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from aegis_trainer.queue.dlq import DeadLetterQueue
from aegis_trainer.queue.overflow import OverflowManager, OverflowStrategy
from aegis_trainer.queue.queue_item import QueueItem

if TYPE_CHECKING:
    from aegis_trainer.utils.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


def _clean_memory() -> None:
    """Clean RAM and VRAM. Standalone implementation matching AirLLM's clean_memory().

    Calls gc.collect(), attempts libc malloc_trim on Linux, and clears
    the CUDA/XPU cache if available.
    """
    gc.collect()

    # Attempt libc malloc_trim on Linux to release freed memory back to OS
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass  # Not on Linux or libc not available

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Intel XPU (Arc GPUs) support
    try:
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except Exception:
        pass


@dataclass
class QQMSConfig:
    """Configuration for the QQMS queue system.

    Attributes:
        max_queue_size: Maximum items that can be queued simultaneously.
        max_attempts: Default maximum retry attempts per item.
        age_boost: Priority boost per second of waiting (subtracted from
            effective priority to increase urgency).
        overflow_strategy: What to do when queue is full.
            One of: "drop_lowest", "block", "spill_disk".
        throttle_check_interval: Seconds between resource threshold checks
            when throttled.
        max_throttle_wait: Maximum seconds to wait when resource-throttled
            before raising an error.
        spill_dir: Directory for disk spillover (only used with spill_disk).
    """

    max_queue_size: int = 256
    max_attempts: int = 3
    age_boost: float = 0.01
    overflow_strategy: str = "drop_lowest"
    throttle_check_interval: float = 5.0
    max_throttle_wait: float = 300.0
    spill_dir: str | None = None


@dataclass
class QQMSStats:
    """Runtime statistics for queue execution.

    Tracks success/failure counts and cumulative execution time for
    performance monitoring and progress reporting.
    """

    items_processed: int = 0
    items_failed: int = 0
    items_dlq: int = 0
    items_retried: int = 0
    total_time: float = 0.0
    total_throttle_time: float = 0.0
    overflow_drops: int = 0

    def record_success(self, elapsed: float) -> None:
        """Record a successful operation execution.

        Args:
            elapsed: Wall-clock seconds the operation took.
        """
        self.items_processed += 1
        self.total_time += elapsed

    def record_failure(self) -> None:
        """Record a failed operation attempt."""
        self.items_failed += 1

    def record_dlq(self) -> None:
        """Record an item sent to the dead letter queue."""
        self.items_dlq += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.items_retried += 1

    def record_throttle(self, wait_time: float) -> None:
        """Record time spent waiting due to resource throttling."""
        self.total_throttle_time += wait_time

    @property
    def avg_time_per_item(self) -> float:
        """Average execution time per successfully processed item."""
        if self.items_processed == 0:
            return 0.0
        return self.total_time / self.items_processed

    @property
    def success_rate(self) -> float:
        """Success rate as a fraction (0.0 to 1.0)."""
        total = self.items_processed + self.items_dlq
        if total == 0:
            return 1.0
        return self.items_processed / total

    def summary(self) -> str:
        """Human-readable summary of queue statistics."""
        lines = [
            f"QQMS Stats:",
            f"  Processed: {self.items_processed}",
            f"  Failed attempts: {self.items_failed}",
            f"  Retried: {self.items_retried}",
            f"  Dead-lettered: {self.items_dlq}",
            f"  Overflow drops: {self.overflow_drops}",
            f"  Total exec time: {self.total_time:.2f}s",
            f"  Avg time/item: {self.avg_time_per_item:.3f}s",
            f"  Throttle wait: {self.total_throttle_time:.2f}s",
            f"  Success rate: {self.success_rate:.1%}",
        ]
        return "\n".join(lines)


class QQMSQueue:
    """Quantized Queue Memory Streaming — resource-aware operation executor.

    The QQMS queue manages the execution of layer modification operations
    with resource monitoring, automatic retry, and dead letter queue routing.
    It is designed for CPU-side weight modification where the full model
    cannot fit in memory.

    Usage:
        config = QQMSConfig(max_queue_size=128)
        queue = QQMSQueue(config)
        queue.bind_resource_monitor(monitor)

        item = QueueItem(
            operation=my_op,
            state_dict=layer_weights,
            context=layer_ctx,
            priority=1.0,
        )
        modified_weights = queue.execute(item)

    Args:
        config: Queue configuration. Uses defaults if None.
    """

    def __init__(self, config: QQMSConfig | None = None) -> None:
        self._config = config or QQMSConfig()
        self._stats = QQMSStats()
        self._dlq = DeadLetterQueue(max_size=100)
        self._resource_monitor: ResourceMonitor | None = None
        self._pending: list[QueueItem] = []

        # Map string strategy to enum
        strategy_map = {
            "drop_lowest": OverflowStrategy.DROP_LOWEST,
            "block": OverflowStrategy.BLOCK,
            "spill_disk": OverflowStrategy.SPILL_DISK,
        }
        strategy = strategy_map.get(
            self._config.overflow_strategy,
            OverflowStrategy.DROP_LOWEST,
        )
        self._overflow = OverflowManager(
            strategy=strategy,
            max_queue_size=self._config.max_queue_size,
            spill_dir=self._config.spill_dir,
        )

    @property
    def dlq(self) -> DeadLetterQueue:
        """Access the dead letter queue."""
        return self._dlq

    @property
    def pending_count(self) -> int:
        """Number of items currently pending in the queue."""
        return len(self._pending)

    def bind_resource_monitor(self, monitor: ResourceMonitor) -> None:
        """Bind a resource monitor for throttle-aware execution.

        When bound, the queue will pause execution when system resources
        exceed configured thresholds (RAM, VRAM, CPU).

        Args:
            monitor: ResourceMonitor instance to check thresholds against.
        """
        self._resource_monitor = monitor
        logger.info("QQMS: Resource monitor bound")

    def enqueue(self, item: QueueItem) -> QueueItem | None:
        """Add an item to the pending queue.

        Handles overflow if queue is at capacity according to the
        configured overflow strategy.

        Args:
            item: QueueItem to enqueue.

        Returns:
            The dropped item if overflow occurred, None otherwise.
        """
        if len(self._pending) >= self._config.max_queue_size:
            dropped = self._overflow.handle(self._pending, item)
            if dropped is item:
                # New item was dropped — don't add it
                self._stats.overflow_drops += 1
                logger.warning("QQMS: Item dropped by overflow: %s", item)
                return dropped
            elif dropped is not None:
                self._stats.overflow_drops += 1
                logger.info("QQMS: Overflow evicted: %s", dropped)

        self._pending.append(item)
        # Sort by effective priority (lowest = most urgent first)
        self._pending.sort(key=lambda x: x.effective_priority)
        self._overflow.notify_space_available()
        return None

    def execute(self, item: QueueItem) -> dict[str, torch.Tensor]:
        """Execute a single queue item with resource-aware throttling and retry.

        This is the core execution method. It:
        1. Waits if resources are over threshold (throttle loop)
        2. Calls operation.apply(state_dict, context)
        3. On success: records stats, returns modified state_dict
        4. On failure: retries up to max_attempts, then sends to DLQ
        5. Cleans memory after each attempt

        Args:
            item: The QueueItem to execute.

        Returns:
            The modified state_dict after the operation is applied.

        Raises:
            RuntimeError: If the item exhausts all retries and lands in DLQ.
        """
        op_name = type(item.operation).__name__
        layer_name = getattr(item.context, 'layer_name', '?')

        logger.debug(
            "QQMS execute: %s on %s (attempt %d/%d)",
            op_name, layer_name, item.attempt_count + 1, item.max_attempts,
        )

        last_error: Exception | None = None

        while not item.is_exhausted:
            # Step 1: Resource throttling
            self._wait_for_resources(op_name, layer_name)

            # Step 2: Execute the operation
            start_time = time.time()
            try:
                result = item.operation.apply(item.state_dict, item.context)
                elapsed = time.time() - start_time

                # Step 3: Success
                self._stats.record_success(elapsed)
                logger.info(
                    "QQMS success: %s on %s in %.3fs",
                    op_name, layer_name, elapsed,
                )

                # Step 5: Clean memory after execution
                _clean_memory()

                return result

            except Exception as exc:
                elapsed = time.time() - start_time
                last_error = exc
                item.increment_attempt()
                self._stats.record_failure()

                logger.warning(
                    "QQMS failure: %s on %s (attempt %d/%d) — %s",
                    op_name, layer_name,
                    item.attempt_count, item.max_attempts,
                    exc,
                )

                # Step 5: Clean memory after each attempt
                _clean_memory()

                if not item.is_exhausted:
                    self._stats.record_retry()
                    # Brief pause before retry (exponential backoff)
                    backoff = min(2.0 ** (item.attempt_count - 1), 30.0)
                    logger.info(
                        "QQMS retry: waiting %.1fs before attempt %d",
                        backoff, item.attempt_count + 1,
                    )
                    time.sleep(backoff)

        # Step 4: Exhausted retries — send to DLQ
        error_str = str(last_error) if last_error else "Unknown error"
        reason = f"Exhausted {item.max_attempts} attempts: {error_str}"
        self.send_to_dlq_from_item(item, reason)

        raise RuntimeError(
            f"QQMS: {op_name} on {layer_name} failed after "
            f"{item.max_attempts} attempts: {error_str}"
        )

    def execute_pending(self) -> list[dict[str, torch.Tensor]]:
        """Execute all pending items in priority order.

        Items are consumed from the pending queue in order of effective
        priority (most urgent first). Failed items that exhaust retries
        are sent to the DLQ.

        Returns:
            List of modified state_dicts for successful operations.
        """
        results: list[dict[str, torch.Tensor]] = []

        while self._pending:
            # Re-sort by effective priority (accounts for aging)
            self._pending.sort(key=lambda x: x.effective_priority)
            item = self._pending.pop(0)
            self._overflow.notify_space_available()

            try:
                result = self.execute(item)
                results.append(result)
            except RuntimeError:
                # Item was sent to DLQ, continue with remaining items
                continue

        return results

    def send_to_dlq(
        self,
        ctx: Any,
        op: Any,
        reason: str,
        error: str | None = None,
    ) -> None:
        """Send a failed operation to the dead letter queue.

        Used when failure occurs outside the normal execute() path.

        Args:
            ctx: LayerContext of the failed operation.
            op: LayerOperation that failed.
            reason: Human-readable failure description.
            error: Optional exception string.
        """
        self._dlq.add_from_context(ctx, op, reason, error=error)
        self._stats.record_dlq()

    def send_to_dlq_from_item(self, item: QueueItem, reason: str) -> None:
        """Send a QueueItem to the dead letter queue.

        Args:
            item: The exhausted QueueItem.
            reason: Failure reason.
        """
        self._dlq.add(item, reason)
        self._stats.record_dlq()

    def get_stats(self) -> QQMSStats:
        """Return current queue execution statistics.

        Returns:
            QQMSStats dataclass with current counters.
        """
        return self._stats

    def get_dlq_report(self) -> list:
        """Return a report of all dead letter queue entries.

        Returns:
            List of DLQEntry objects.
        """
        return self._dlq.get_report()

    def _wait_for_resources(self, op_name: str, layer_name: str) -> None:
        """Block until system resources are below threshold.

        If no resource monitor is bound, returns immediately.
        Uses the configured throttle_check_interval and max_throttle_wait.

        Args:
            op_name: Operation name (for logging).
            layer_name: Layer name (for logging).

        Raises:
            TimeoutError: If resource throttling exceeds max_throttle_wait.
        """
        if self._resource_monitor is None:
            return

        if not self._resource_monitor.is_over_threshold():
            return

        logger.warning(
            "QQMS throttle: resources over threshold, pausing %s on %s",
            op_name, layer_name,
        )

        throttle_start = time.time()

        while self._resource_monitor.is_over_threshold():
            elapsed = time.time() - throttle_start

            if elapsed >= self._config.max_throttle_wait:
                self._stats.record_throttle(elapsed)
                raise TimeoutError(
                    f"QQMS: Resource throttle timeout after {elapsed:.1f}s "
                    f"while waiting to execute {op_name} on {layer_name}. "
                    f"System resources remain over threshold."
                )

            # Clean memory to try to free resources
            _clean_memory()

            logger.debug(
                "QQMS throttle: waiting %.1fs/%.1fs for resources",
                elapsed, self._config.max_throttle_wait,
            )

            time.sleep(self._config.throttle_check_interval)

        total_wait = time.time() - throttle_start
        self._stats.record_throttle(total_wait)

        logger.info(
            "QQMS throttle: resources available after %.1fs wait",
            total_wait,
        )

    def reset_stats(self) -> None:
        """Reset all queue statistics to zero."""
        self._stats = QQMSStats()

    def __repr__(self) -> str:
        return (
            f"QQMSQueue(pending={self.pending_count}, "
            f"processed={self._stats.items_processed}, "
            f"dlq={self._dlq.size}, "
            f"config={self._config})"
        )
