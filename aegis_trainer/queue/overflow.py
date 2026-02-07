"""
Overflow Manager — Handles queue capacity overflow with configurable strategies.

When the QQMS queue reaches its maximum size, the overflow manager decides
what to do with new items: drop the lowest priority item, block until space
is available, or spill to disk.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aegis_trainer.queue.queue_item import QueueItem

logger = logging.getLogger(__name__)


class OverflowStrategy(Enum):
    """Strategy for handling queue overflow when max capacity is reached."""

    DROP_LOWEST = "drop_lowest"   # Drop the lowest-priority (highest value) item
    BLOCK = "block"               # Block until space becomes available
    SPILL_DISK = "spill_disk"     # Serialize overflow items to disk


class OverflowManager:
    """Manages queue overflow using a configurable strategy.

    When the queue is at capacity and a new item needs to be enqueued,
    this manager applies the configured strategy to make room.

    Args:
        strategy: The overflow handling strategy to use.
        max_queue_size: Maximum number of items the queue can hold.
        block_timeout: Maximum seconds to wait when using BLOCK strategy.
        spill_dir: Directory for disk spillover (auto-created if needed).
    """

    def __init__(
        self,
        strategy: OverflowStrategy = OverflowStrategy.DROP_LOWEST,
        max_queue_size: int = 256,
        block_timeout: float = 60.0,
        spill_dir: str | Path | None = None,
    ) -> None:
        self._strategy = strategy
        self._max_queue_size = max_queue_size
        self._block_timeout = block_timeout
        self._spill_dir = Path(spill_dir) if spill_dir else None
        self._space_available = threading.Event()
        self._space_available.set()  # Initially, space is available
        self._spilled_count = 0
        self._dropped_count = 0
        self._lock = threading.Lock()

    @property
    def strategy(self) -> OverflowStrategy:
        """Current overflow strategy."""
        return self._strategy

    @property
    def dropped_count(self) -> int:
        """Total number of items dropped due to overflow."""
        return self._dropped_count

    @property
    def spilled_count(self) -> int:
        """Total number of items spilled to disk."""
        return self._spilled_count

    def handle(
        self,
        queue: list[QueueItem],
        new_item: QueueItem,
    ) -> QueueItem | None:
        """Handle overflow when the queue is at capacity.

        This method is called when the queue has reached max_queue_size
        and a new item needs to be added.

        Args:
            queue: The current queue (mutable list of QueueItems).
            new_item: The item attempting to be enqueued.

        Returns:
            The dropped/evicted QueueItem, or None if no item was dropped
            (e.g., the new item was spilled to disk or blocking resolved).
        """
        if len(queue) < self._max_queue_size:
            # No overflow — caller should just append
            return None

        if self._strategy == OverflowStrategy.DROP_LOWEST:
            return self._handle_drop_lowest(queue, new_item)
        elif self._strategy == OverflowStrategy.BLOCK:
            return self._handle_block(queue, new_item)
        elif self._strategy == OverflowStrategy.SPILL_DISK:
            return self._handle_spill_disk(queue, new_item)
        else:
            raise ValueError(f"Unknown overflow strategy: {self._strategy}")

    def _handle_drop_lowest(
        self,
        queue: list[QueueItem],
        new_item: QueueItem,
    ) -> QueueItem | None:
        """Drop the lowest-priority item (highest effective_priority value).

        If the new item is lower priority than everything in the queue,
        the new item itself is dropped.
        """
        if not queue:
            return None

        # Find the item with the highest effective_priority value
        # (i.e., least urgent)
        worst_idx = 0
        worst_priority = queue[0].effective_priority
        for i, item in enumerate(queue[1:], 1):
            ep = item.effective_priority
            if ep > worst_priority:
                worst_priority = ep
                worst_idx = i

        # If the new item is less urgent than the worst in queue, drop new item
        if new_item.effective_priority > worst_priority:
            logger.info(
                "Overflow DROP_LOWEST: new item %s is lower priority than "
                "all queued items, dropping it",
                new_item,
            )
            with self._lock:
                self._dropped_count += 1
            return new_item

        # Otherwise, evict the worst item and let the caller add new_item
        dropped = queue.pop(worst_idx)
        logger.info(
            "Overflow DROP_LOWEST: evicted %s to make room for %s",
            dropped, new_item,
        )
        with self._lock:
            self._dropped_count += 1
        return dropped

    def _handle_block(
        self,
        queue: list[QueueItem],
        new_item: QueueItem,
    ) -> QueueItem | None:
        """Block until space becomes available or timeout is reached.

        If timeout is reached, falls back to DROP_LOWEST behavior.
        """
        self._space_available.clear()

        logger.info(
            "Overflow BLOCK: queue full (%d items), waiting up to %.1fs",
            len(queue), self._block_timeout,
        )

        start = time.time()
        while len(queue) >= self._max_queue_size:
            elapsed = time.time() - start
            if elapsed >= self._block_timeout:
                logger.warning(
                    "Overflow BLOCK: timeout after %.1fs, falling back to DROP_LOWEST",
                    elapsed,
                )
                return self._handle_drop_lowest(queue, new_item)

            # Wait briefly then recheck
            self._space_available.wait(timeout=0.5)
            if self._space_available.is_set():
                break

        return None  # Space is now available

    def _handle_spill_disk(
        self,
        queue: list[QueueItem],
        new_item: QueueItem,
    ) -> QueueItem | None:
        """Serialize the least-urgent item to disk to make room.

        The spilled item's metadata is saved (operation name, context info,
        priority), though full tensor state is not preserved (the layer
        can be reloaded from the model on disk).
        """
        spill_dir = self._spill_dir or Path(tempfile.mkdtemp(prefix="qqms_spill_"))
        spill_dir.mkdir(parents=True, exist_ok=True)

        if not queue:
            return None

        # Find the least-urgent item
        worst_idx = 0
        worst_priority = queue[0].effective_priority
        for i, item in enumerate(queue[1:], 1):
            ep = item.effective_priority
            if ep > worst_priority:
                worst_priority = ep
                worst_idx = i

        spilled = queue.pop(worst_idx)

        # Save metadata (not tensors — those can be reloaded from model files)
        spill_file = spill_dir / f"spill_{self._spilled_count:06d}.json"
        spill_meta = {
            'operation': type(spilled.operation).__name__,
            'layer_name': getattr(spilled.context, 'layer_name', ''),
            'priority': spilled.priority,
            'effective_priority': spilled.effective_priority,
            'created_at': spilled.created_at,
            'attempt_count': spilled.attempt_count,
            'spilled_at': time.time(),
        }

        try:
            with open(spill_file, 'w') as f:
                json.dump(spill_meta, f, indent=2)
            logger.info(
                "Overflow SPILL_DISK: spilled %s to %s",
                spilled, spill_file,
            )
        except OSError as exc:
            logger.error(
                "Overflow SPILL_DISK: failed to write %s — %s, "
                "falling back to DROP_LOWEST",
                spill_file, exc,
            )
            with self._lock:
                self._dropped_count += 1
            return spilled

        with self._lock:
            self._spilled_count += 1

        return spilled

    def notify_space_available(self) -> None:
        """Signal that space has become available in the queue.

        Called by the queue after an item is consumed, to unblock
        any thread waiting in BLOCK strategy.
        """
        self._space_available.set()

    def get_stats(self) -> dict:
        """Return overflow statistics."""
        return {
            'strategy': self._strategy.value,
            'max_queue_size': self._max_queue_size,
            'dropped_count': self._dropped_count,
            'spilled_count': self._spilled_count,
        }

    def __repr__(self) -> str:
        return (
            f"OverflowManager(strategy={self._strategy.value}, "
            f"max_size={self._max_queue_size}, "
            f"dropped={self._dropped_count}, "
            f"spilled={self._spilled_count})"
        )
