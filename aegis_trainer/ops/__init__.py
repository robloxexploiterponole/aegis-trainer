"""
Layer-level operations for AEGIS AI Trainer.

All concrete operations inherit from LayerOperation and implement
should_apply() + apply() for per-layer weight modification.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from .abliteration import AbliterationOp
from .base import LayerOperation
from .expert_prune import ExpertPruneOp
from .longrope import LongRoPEOp
from .lora_merge import LoRAMergeOp
from .quantize import QuantizeOp
from .weight_inspect import WeightInspectOp

__all__ = [
    "LayerOperation",
    "AbliterationOp",
    "LongRoPEOp",
    "LoRAMergeOp",
    "WeightInspectOp",
    "ExpertPruneOp",
    "QuantizeOp",
]
