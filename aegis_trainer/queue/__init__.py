"""
QQMS — Quantized Queue Memory Streaming queue system.

Provides the priority queue, overflow management, dead letter queue,
and resource-aware execution engine for layer-by-layer model modification.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from aegis_trainer.queue.dlq import DeadLetterQueue, DLQEntry
from aegis_trainer.queue.overflow import OverflowManager, OverflowStrategy
from aegis_trainer.queue.qqms import QQMSConfig, QQMSQueue, QQMSStats
from aegis_trainer.queue.queue_item import QueueItem

__all__ = [
    "QQMSQueue",
    "QQMSConfig",
    "QQMSStats",
    "QueueItem",
    "DeadLetterQueue",
    "DLQEntry",
    "OverflowManager",
    "OverflowStrategy",
]
