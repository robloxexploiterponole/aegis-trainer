"""
LayerOperation — Abstract base class for all AEGIS layer-level operations.

Every concrete operation (abliteration, LongRoPE, LoRA merge, etc.) inherits
from LayerOperation and implements should_apply() + apply(). The trainer
orchestrator calls these in sequence for each layer.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict

import torch

from ..layer_context import LayerContext

logger = logging.getLogger(__name__)


class LayerOperation(ABC):
    """Abstract base class for per-layer weight modification operations.

    Subclasses must implement:
        - should_apply(ctx): Decide if this operation targets the given layer.
        - apply(state_dict, ctx): Modify and return the layer's state dict.

    Optionally override:
        - validate(original, modified, ctx): Sanity-check the modification.
        - estimate_memory(ctx): Estimate peak memory in bytes for apply().
    """

    name: str = "unnamed_operation"

    @abstractmethod
    def should_apply(self, ctx: LayerContext) -> bool:
        """Return True if this operation should apply to this layer.

        Args:
            ctx: LayerContext with layer metadata (index, type, architecture params).

        Returns:
            True if the operation should be applied to this layer.
        """

    @abstractmethod
    def apply(
        self,
        state_dict: dict[str, torch.Tensor],
        ctx: LayerContext,
    ) -> dict[str, torch.Tensor]:
        """Apply the operation to the layer's state dict.

        Must not modify the input state_dict in place if validate() will
        compare original vs modified. Return a new dict or mutate and return.

        Args:
            state_dict: Mapping of parameter names to tensors for this layer.
            ctx: LayerContext with layer metadata.

        Returns:
            Modified state dict (may be the same object if mutated in place).

        Raises:
            RuntimeError: If the operation fails and should abort this layer.
        """

    def validate(
        self,
        original: dict[str, torch.Tensor],
        modified: dict[str, torch.Tensor],
        ctx: LayerContext,
    ) -> bool:
        """Optional post-apply validation.

        Called after apply() to sanity-check the result. Default implementation
        checks that no tensor contains NaN or Inf values.

        Args:
            original: The state dict before apply() (may be empty if not preserved).
            modified: The state dict returned by apply().
            ctx: LayerContext with layer metadata.

        Returns:
            True if modification looks correct, False if something is wrong.
        """
        for key, tensor in modified.items():
            if torch.isnan(tensor).any():
                logger.error(
                    "Validation failed: NaN detected in %s for layer %s",
                    key,
                    ctx.layer_name,
                )
                return False
            if torch.isinf(tensor).any():
                logger.error(
                    "Validation failed: Inf detected in %s for layer %s",
                    key,
                    ctx.layer_name,
                )
                return False
        return True

    def estimate_memory(self, ctx: LayerContext) -> int:
        """Estimate peak memory usage in bytes for apply() on this layer.

        Used by the resource monitor to decide whether to proceed or throttle.
        Default returns 0 (unknown).

        Args:
            ctx: LayerContext with layer metadata.

        Returns:
            Estimated peak memory in bytes, or 0 if unknown.
        """
        return 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
