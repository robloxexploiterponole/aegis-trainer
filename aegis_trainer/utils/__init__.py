"""
Utility modules for AEGIS AI Trainer.

Provides resource monitoring, layer I/O, checkpoint management,
and profiling infrastructure.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from .checkpoint import CheckpointManager
from .layer_io import LayerIO
from .profiler import TrainerProfiler
from .resource_monitor import ResourceLimits, ResourceMonitor, ResourceSnapshot

__all__ = [
    "CheckpointManager",
    "LayerIO",
    "ResourceLimits",
    "ResourceMonitor",
    "ResourceSnapshot",
    "TrainerProfiler",
]
