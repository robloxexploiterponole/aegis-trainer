"""
AEGIS AI Trainer — Layer-by-layer model training and modification framework.

Uses AirLLM layer streaming to modify 80B+ parameter MoE models on consumer
GPUs without ever loading the full model into memory.

Operations:
  - Abliteration (deregulation via directional ablation)
  - LongRoPE (context window extension)
  - LoRA merge (adapter integration into base weights)
  - Per-layer weight inspection and modification

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

__version__ = "0.1.0"
__author__ = "Hardwick Software Services"
__license__ = "SSPL-1.0"

from aegis_trainer.layer_context import LayerContext
from aegis_trainer.layer_trainer import LayerTrainer, ProgressUpdate, TrainerResult
from aegis_trainer.ops.base import LayerOperation
from aegis_trainer.queue import (
    DeadLetterQueue,
    DLQEntry,
    OverflowManager,
    OverflowStrategy,
    QQMSConfig,
    QQMSQueue,
    QQMSStats,
    QueueItem,
)
from aegis_trainer.utils import (
    CheckpointManager,
    LayerIO,
    ResourceLimits,
    ResourceMonitor,
    ResourceSnapshot,
    TrainerProfiler,
)

__all__ = [
    # Core
    "LayerTrainer",
    "LayerContext",
    "LayerOperation",
    "ProgressUpdate",
    "TrainerResult",
    # Queue (QQMS)
    "QQMSQueue",
    "QQMSConfig",
    "QQMSStats",
    "QueueItem",
    "DeadLetterQueue",
    "DLQEntry",
    "OverflowManager",
    "OverflowStrategy",
    # Utils
    "CheckpointManager",
    "LayerIO",
    "ResourceLimits",
    "ResourceMonitor",
    "ResourceSnapshot",
    "TrainerProfiler",
]
