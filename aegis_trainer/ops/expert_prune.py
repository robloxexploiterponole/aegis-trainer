"""
Expert pruning — zero out or scale specific experts in MoE layers.

Provides targeted expert removal or downscaling for Mixture-of-Experts
layers. Can zero out entire experts (effectively removing them from the
mixture) or scale their contribution by a specified factor.

Useful for:
  - Removing toxic/biased experts identified via activation analysis
  - Reducing model size by pruning least-used experts
  - Experimental expert surgery for capability modification

Qwen3-Next has 512 experts per layer with 10 active per token.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch
from torch import Tensor

from ..layer_context import LayerContext
from .base import LayerOperation

logger = logging.getLogger(__name__)

# Regex for expert weight keys:
# model.layers.{i}.mlp.experts.{N}.{gate,up,down}_proj.weight
_EXPERT_KEY_PATTERN = re.compile(
    r"^(?:.*\.)?mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

# Valid pruning modes
_VALID_PRUNE_MODES = {"zero", "scale"}


class ExpertPruneOp(LayerOperation):
    """Expert pruning operation — zero out or scale specific MoE experts.

    Targets individual expert weight tensors (gate_proj, up_proj, down_proj)
    and either zeros them out completely or scales them by a specified factor.

    Args:
        experts_to_prune: List of expert indices to prune (0-based). If None,
            no experts are pruned (operation is a no-op).
        prune_mode: How to prune experts. "zero" replaces all weights with
            zeros. "scale" multiplies all weights by scale_factor.
        scale_factor: Scaling factor for "scale" mode. Default 0.0 (equivalent
            to zeroing). Values between 0 and 1 reduce expert contribution.
        prune_shared_expert: Whether to also prune the shared expert. This is
            almost never desired since the shared expert processes every token.
            Defaults to False.
    """

    name: str = "expert_prune"

    def __init__(
        self,
        experts_to_prune: list[int] | None = None,
        prune_mode: str = "zero",
        scale_factor: float = 0.0,
        prune_shared_expert: bool = False,
    ) -> None:
        if prune_mode not in _VALID_PRUNE_MODES:
            raise ValueError(
                f"Invalid prune_mode: {prune_mode!r}. "
                f"Must be one of {_VALID_PRUNE_MODES}"
            )

        self.experts_to_prune = set(experts_to_prune) if experts_to_prune else set()
        self.prune_mode = prune_mode
        self.scale_factor = scale_factor
        self.prune_shared_expert = prune_shared_expert

        # Stats
        self._tensors_pruned = 0
        self._layers_processed = 0
        self._experts_pruned_per_layer: dict[int, int] = {}

        if self.experts_to_prune:
            logger.info(
                "ExpertPrune initialized: mode=%s, scale=%.4f, "
                "targeting %d experts: %s",
                prune_mode, scale_factor,
                len(self.experts_to_prune),
                sorted(self.experts_to_prune),
            )

    def should_apply(self, ctx: LayerContext) -> bool:
        """Only apply to MoE layers (layers with experts > 0)."""
        return ctx.num_experts > 0 and len(self.experts_to_prune) > 0

    def _prune_tensor(self, tensor: Tensor) -> Tensor:
        """Apply pruning to a single tensor.

        Args:
            tensor: Weight tensor to prune.

        Returns:
            Pruned tensor (zeroed or scaled).
        """
        if self.prune_mode == "zero":
            return torch.zeros_like(tensor)
        elif self.prune_mode == "scale":
            return tensor * self.scale_factor
        else:
            # Should never reach here due to __init__ validation
            return tensor

    def apply(
        self,
        state_dict: dict[str, Tensor],
        ctx: LayerContext,
    ) -> dict[str, Tensor]:
        """Zero out or scale weight tensors for specified experts.

        Targets expert keys matching pattern:
            mlp.experts.{N}.{gate,up,down}_proj.weight

        Args:
            state_dict: Layer state dict containing expert weight tensors.
            ctx: LayerContext with layer metadata.

        Returns:
            Modified state dict with pruned expert weights.
        """
        pruned_count = 0
        experts_seen: set[int] = set()

        for key in list(state_dict.keys()):
            match = _EXPERT_KEY_PATTERN.search(key)
            if not match:
                # Also handle shared expert if configured
                if self.prune_shared_expert and "shared_expert" in key:
                    state_dict[key] = self._prune_tensor(state_dict[key])
                    pruned_count += 1
                    logger.debug(
                        "Pruned shared expert tensor: %s", key
                    )
                continue

            expert_idx = int(match.group(1))

            if expert_idx in self.experts_to_prune:
                state_dict[key] = self._prune_tensor(state_dict[key])
                pruned_count += 1
                experts_seen.add(expert_idx)

                logger.debug(
                    "Pruned expert %d tensor: %s (mode=%s)",
                    expert_idx, key, self.prune_mode,
                )

        self._tensors_pruned += pruned_count
        self._layers_processed += 1
        self._experts_pruned_per_layer[ctx.layer_index] = len(experts_seen)

        logger.info(
            "Pruned %d tensors across %d experts in layer %d (%s, mode=%s)",
            pruned_count,
            len(experts_seen),
            ctx.layer_index,
            ctx.layer_type,
            self.prune_mode,
        )

        return state_dict

    def validate(
        self,
        original: dict[str, Tensor],
        modified: dict[str, Tensor],
        ctx: LayerContext,
    ) -> bool:
        """Validate that pruning was applied correctly."""
        if not super().validate(original, modified, ctx):
            return False

        if self.prune_mode == "zero":
            # Verify that pruned expert tensors are actually zero
            for key in modified:
                match = _EXPERT_KEY_PATTERN.search(key)
                if match:
                    expert_idx = int(match.group(1))
                    if expert_idx in self.experts_to_prune:
                        if modified[key].any():
                            logger.error(
                                "Expert %d tensor %s should be zero but is not.",
                                expert_idx, key,
                            )
                            return False

        return True

    def estimate_memory(self, ctx: LayerContext) -> int:
        """Estimate peak memory: minimal since we just zero/scale in place."""
        # Pruning is nearly free — just tensor operations, no extra copies needed
        # The largest single expert tensor determines peak
        # down_proj: [hidden_size, intermediate_size_per_expert]
        return ctx.hidden_size * 1024 * 4  # Conservative estimate

    @property
    def stats(self) -> dict:
        """Return summary statistics from the pruning run."""
        return {
            "tensors_pruned": self._tensors_pruned,
            "layers_processed": self._layers_processed,
            "experts_targeted": sorted(self.experts_to_prune),
            "prune_mode": self.prune_mode,
            "scale_factor": self.scale_factor,
            "experts_pruned_per_layer": self._experts_pruned_per_layer,
        }
