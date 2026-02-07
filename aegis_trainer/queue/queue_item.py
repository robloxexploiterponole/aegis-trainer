"""
QueueItem — Priority-boosted queue item for QQMS layer operation scheduling.

Items age-boost their priority over time so long-waiting operations get
executed before newly submitted ones, preventing starvation in the
layer-by-layer training pipeline.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from aegis_trainer.ops.base import LayerOperation
    from aegis_trainer.layer_context import LayerContext


@dataclass
class QueueItem:
    """A single queued layer operation with age-boosted priority.

    Priority is dynamic: as an item waits in the queue, its effective
    priority decreases (becomes more urgent). This prevents starvation
    when high-priority items are continuously submitted.

    Attributes:
        operation: The LayerOperation to apply.
        state_dict: Layer weights as tensor dict (loaded from safetensors).
        context: LayerContext describing the target layer.
        priority: Base priority (lower = higher priority / more urgent).
        created_at: Timestamp when the item was enqueued.
        attempt_count: Number of execution attempts so far.
        max_attempts: Maximum retries before sending to dead letter queue.
        age_boost: Priority boost per second of waiting (subtracted from
            effective priority to make it more urgent over time).
    """

    operation: LayerOperation
    state_dict: dict[str, torch.Tensor]
    context: LayerContext
    priority: float
    created_at: float = field(default_factory=time.time)
    attempt_count: int = 0
    max_attempts: int = 3
    age_boost: float = 0.01

    @property
    def effective_priority(self) -> float:
        """Priority decreases (gets more urgent) with age.

        A lower value means higher urgency. The age boost is subtracted
        so items that have been waiting longer are processed first.
        """
        age = time.time() - self.created_at
        return self.priority - (age * self.age_boost)

    @property
    def is_exhausted(self) -> bool:
        """True if this item has exceeded its maximum retry attempts."""
        return self.attempt_count >= self.max_attempts

    def increment_attempt(self) -> None:
        """Record a failed execution attempt."""
        self.attempt_count += 1

    def __lt__(self, other: QueueItem) -> bool:
        """Compare by effective priority for heap ordering."""
        return self.effective_priority < other.effective_priority

    def __repr__(self) -> str:
        op_name = type(self.operation).__name__ if self.operation else "None"
        layer = getattr(self.context, 'layer_name', '?') if self.context else '?'
        return (
            f"QueueItem(op={op_name}, layer={layer}, "
            f"priority={self.priority:.3f}, "
            f"effective={self.effective_priority:.3f}, "
            f"attempts={self.attempt_count}/{self.max_attempts})"
        )
