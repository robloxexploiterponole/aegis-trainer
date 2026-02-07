"""
Dead Letter Queue — Captures failed operations for inspection and retry.

When a QueueItem exhausts its retry budget, it lands here. The DLQ provides
structured reporting so operators can diagnose failures and optionally
retry them in batch.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from aegis_trainer.layer_context import LayerContext
    from aegis_trainer.ops.base import LayerOperation
    from aegis_trainer.queue.queue_item import QueueItem

logger = logging.getLogger(__name__)


@dataclass
class DLQEntry:
    """A single dead letter queue entry recording a failed operation.

    Attributes:
        context: The LayerContext the operation was targeting.
        operation_name: Name/class of the failed operation.
        reason: Human-readable reason for failure.
        timestamp: When the entry was created.
        attempt_count: How many times execution was attempted.
        last_error: String representation of the last exception, if any.
        layer_name: Name of the layer that failed (extracted from context).
        operation_ref: Optional reference to the original operation for retry.
    """

    context: LayerContext
    operation_name: str
    reason: str
    timestamp: float
    attempt_count: int
    last_error: str | None = None
    layer_name: str = ""
    operation_ref: LayerOperation | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.layer_name and self.context is not None:
            self.layer_name = getattr(self.context, 'layer_name', '')

    def to_dict(self) -> dict:
        """Serialize to dictionary for reporting."""
        return {
            'operation': self.operation_name,
            'layer': self.layer_name,
            'reason': self.reason,
            'timestamp': self.timestamp,
            'attempts': self.attempt_count,
            'last_error': self.last_error,
        }


class DeadLetterQueue:
    """Thread-safe dead letter queue for failed layer operations.

    Collects failed operations with full context for post-run diagnosis.
    Supports batch retry with a caller-provided executor function.

    Args:
        max_size: Maximum entries to retain. When exceeded, oldest entries
            are silently dropped (FIFO eviction).
    """

    def __init__(self, max_size: int = 100) -> None:
        self._entries: deque[DLQEntry] = deque(maxlen=max_size)
        self._max_size = max_size
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Current number of entries in the DLQ."""
        with self._lock:
            return len(self._entries)

    def add(self, item: QueueItem, reason: str) -> None:
        """Add a failed QueueItem to the dead letter queue.

        Args:
            item: The QueueItem that exhausted its retries.
            reason: Human-readable explanation of why it failed.
        """
        last_error = None
        op_name = type(item.operation).__name__ if item.operation else "Unknown"

        entry = DLQEntry(
            context=item.context,
            operation_name=op_name,
            reason=reason,
            timestamp=time.time(),
            attempt_count=item.attempt_count,
            last_error=last_error,
            operation_ref=item.operation,
        )

        with self._lock:
            self._entries.append(entry)

        logger.warning(
            "DLQ: %s on layer %s — %s (attempts: %d)",
            op_name, entry.layer_name, reason, item.attempt_count,
        )

    def add_from_context(
        self,
        ctx: LayerContext,
        op: LayerOperation,
        reason: str,
        error: str | None = None,
    ) -> None:
        """Add a failed operation directly from context and operation refs.

        This is used when failure occurs outside the normal queue execution
        path (e.g., during validation or save).

        Args:
            ctx: The LayerContext of the failed operation.
            op: The LayerOperation that failed.
            reason: Human-readable failure description.
            error: Optional exception string.
        """
        op_name = type(op).__name__ if op else "Unknown"

        entry = DLQEntry(
            context=ctx,
            operation_name=op_name,
            reason=reason,
            timestamp=time.time(),
            attempt_count=0,
            last_error=error,
            operation_ref=op,
        )

        with self._lock:
            self._entries.append(entry)

        logger.warning(
            "DLQ (direct): %s on layer %s — %s",
            op_name, entry.layer_name, reason,
        )

    def get_report(self) -> list[DLQEntry]:
        """Return a snapshot of all DLQ entries.

        Returns:
            List of DLQEntry objects, oldest first.
        """
        with self._lock:
            return list(self._entries)

    def get_summary(self) -> dict:
        """Return a summary of DLQ state for logging/display.

        Returns:
            Dict with counts by operation type and total.
        """
        with self._lock:
            entries = list(self._entries)

        by_op: dict[str, int] = {}
        for entry in entries:
            by_op[entry.operation_name] = by_op.get(entry.operation_name, 0) + 1

        return {
            'total': len(entries),
            'by_operation': by_op,
            'entries': [e.to_dict() for e in entries],
        }

    def retry_all(
        self,
        executor_fn: Callable[[LayerContext, LayerOperation], bool],
    ) -> list[bool]:
        """Attempt to retry all DLQ entries using the provided executor.

        The executor_fn receives (context, operation) and should return
        True on success, False on failure. Successfully retried entries
        are removed from the DLQ.

        Args:
            executor_fn: Callable that attempts to re-execute the operation.
                Signature: (LayerContext, LayerOperation) -> bool

        Returns:
            List of booleans indicating success/failure for each entry.
        """
        with self._lock:
            entries = list(self._entries)

        results: list[bool] = []
        succeeded_indices: list[int] = []

        for i, entry in enumerate(entries):
            if entry.operation_ref is None:
                logger.warning(
                    "DLQ retry: no operation ref for %s on %s, skipping",
                    entry.operation_name, entry.layer_name,
                )
                results.append(False)
                continue

            try:
                success = executor_fn(entry.context, entry.operation_ref)
                results.append(success)
                if success:
                    succeeded_indices.append(i)
                    logger.info(
                        "DLQ retry succeeded: %s on %s",
                        entry.operation_name, entry.layer_name,
                    )
                else:
                    logger.warning(
                        "DLQ retry failed: %s on %s",
                        entry.operation_name, entry.layer_name,
                    )
            except Exception as exc:
                logger.error(
                    "DLQ retry exception: %s on %s — %s",
                    entry.operation_name, entry.layer_name, exc,
                )
                results.append(False)

        # Remove successfully retried entries (in reverse to preserve indices)
        if succeeded_indices:
            with self._lock:
                remaining = [
                    e for i, e in enumerate(self._entries)
                    if i not in set(succeeded_indices)
                ]
                self._entries.clear()
                self._entries.extend(remaining)

        return results

    def clear(self) -> int:
        """Clear all entries from the DLQ.

        Returns:
            Number of entries that were cleared.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def __len__(self) -> int:
        return self.size

    def __bool__(self) -> bool:
        return self.size > 0

    def __repr__(self) -> str:
        return f"DeadLetterQueue(entries={self.size}, max_size={self._max_size})"
