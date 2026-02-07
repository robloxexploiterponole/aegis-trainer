"""
LoRA adapter merge — bakes LoRA weights directly into base model weights.

Merges Low-Rank Adaptation (LoRA) adapter weights into the base model by
computing W_merged = W_original + (lora_B @ lora_A) * scaling, where
scaling = lora_alpha / lora_rank.

Supports both DeltaNet and full_attention layers. Handles PEFT naming
conventions for LoRA keys and maps them to base model weight keys.

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

# PEFT key patterns for LoRA adapters
# base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight
# base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight
_PEFT_LORA_A_PATTERN = re.compile(
    r"^(?:base_model\.model\.)?(?:model\.)?layers\.(\d+)\.(.+)\.lora_A\.(?:default\.)?weight$"
)
_PEFT_LORA_B_PATTERN = re.compile(
    r"^(?:base_model\.model\.)?(?:model\.)?layers\.(\d+)\.(.+)\.lora_B\.(?:default\.)?weight$"
)

# Base model key format
# model.layers.{i}.self_attn.q_proj.weight
_BASE_KEY_TEMPLATE = "model.layers.{layer_idx}.{module_path}.weight"


def _extract_lora_pairs(
    lora_state_dict: dict[str, Tensor],
    layer_index: int,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Extract paired LoRA A/B weights for a specific layer.

    Scans the full LoRA state dict for keys matching the given layer index
    and pairs up the corresponding lora_A and lora_B weight matrices.

    Args:
        lora_state_dict: Full LoRA adapter state dict.
        layer_index: Layer index to extract pairs for.

    Returns:
        Dict mapping module_path (e.g. "self_attn.q_proj") to
        (lora_A_weight, lora_B_weight) tuples.
    """
    lora_a_weights: dict[str, Tensor] = {}
    lora_b_weights: dict[str, Tensor] = {}

    for key, tensor in lora_state_dict.items():
        match_a = _PEFT_LORA_A_PATTERN.match(key)
        if match_a and int(match_a.group(1)) == layer_index:
            module_path = match_a.group(2)
            lora_a_weights[module_path] = tensor
            continue

        match_b = _PEFT_LORA_B_PATTERN.match(key)
        if match_b and int(match_b.group(1)) == layer_index:
            module_path = match_b.group(2)
            lora_b_weights[module_path] = tensor
            continue

    # Pair up A and B weights
    pairs: dict[str, tuple[Tensor, Tensor]] = {}
    for module_path in lora_a_weights:
        if module_path in lora_b_weights:
            pairs[module_path] = (
                lora_a_weights[module_path],
                lora_b_weights[module_path],
            )
        else:
            logger.warning(
                "Found lora_A for layers.%d.%s but no matching lora_B. Skipping.",
                layer_index, module_path,
            )

    for module_path in lora_b_weights:
        if module_path not in lora_a_weights:
            logger.warning(
                "Found lora_B for layers.%d.%s but no matching lora_A. Skipping.",
                layer_index, module_path,
            )

    return pairs


def _find_base_key(
    state_dict: dict[str, Tensor],
    module_path: str,
) -> str | None:
    """Find the base model key in the state dict for a given module path.

    Tries multiple naming conventions since state_dict keys may or may not
    include the full "model.layers.{i}." prefix (the prefix is typically
    stripped by the layer streaming loader).

    Args:
        state_dict: The layer's state dict.
        module_path: Module path like "self_attn.q_proj".

    Returns:
        Matching key from state_dict, or None if not found.
    """
    # Try exact match with module_path.weight
    target = f"{module_path}.weight"
    if target in state_dict:
        return target

    # Try with various prefixes that might be in the state dict
    for key in state_dict:
        if key.endswith(target):
            return key

    return None


class LoRAMergeOp(LayerOperation):
    """LoRA adapter merge operation — bakes LoRA weights into base model.

    Computes W_merged = W_original + (lora_B @ lora_A) * scaling for each
    base weight that has a corresponding LoRA adapter pair.

    Applies to both DeltaNet (linear_attention) and full_attention layers
    if the adapter has weights for that layer.

    Args:
        lora_state_dict: Full LoRA adapter state dict containing all lora_A
            and lora_B weight tensors.
        lora_alpha: LoRA alpha parameter (scaling numerator). Default 32.
        lora_rank: LoRA rank (scaling denominator). Default 16.
        scaling: Explicit scaling factor. If None, computed as alpha/rank.
        target_modules: Optional list of module names to restrict merging to
            (e.g. ["q_proj", "v_proj"]). If None, merges all available pairs.
    """

    name: str = "lora_merge"

    def __init__(
        self,
        lora_state_dict: dict[str, Tensor],
        lora_alpha: int = 32,
        lora_rank: int = 16,
        scaling: float | None = None,
        target_modules: list[str] | None = None,
    ) -> None:
        self.lora_state_dict = lora_state_dict
        self.lora_alpha = lora_alpha
        self.lora_rank = lora_rank
        self.scaling = scaling if scaling is not None else lora_alpha / lora_rank
        self.target_modules = target_modules

        # Pre-compute which layers have LoRA weights
        self._layer_indices_with_lora: set[int] = set()
        for key in lora_state_dict:
            match = _PEFT_LORA_A_PATTERN.match(key) or _PEFT_LORA_B_PATTERN.match(key)
            if match:
                self._layer_indices_with_lora.add(int(match.group(1)))

        # Stats
        self._tensors_merged = 0
        self._layers_processed = 0

        logger.info(
            "LoRA merge initialized: alpha=%d, rank=%d, scaling=%.4f, "
            "layers with adapters: %d, target_modules=%s",
            lora_alpha, lora_rank, self.scaling,
            len(self._layer_indices_with_lora),
            target_modules,
        )

    def should_apply(self, ctx: LayerContext) -> bool:
        """Apply to any layer that has corresponding LoRA weights."""
        return ctx.layer_index in self._layer_indices_with_lora

    def _should_merge_module(self, module_path: str) -> bool:
        """Check if a module should be merged based on target_modules filter."""
        if self.target_modules is None:
            return True
        # Check if any target module name appears at the end of the module path
        module_name = module_path.split(".")[-1]
        return module_name in self.target_modules

    def apply(
        self,
        state_dict: dict[str, Tensor],
        ctx: LayerContext,
    ) -> dict[str, Tensor]:
        """Merge LoRA adapter weights into base model weights for this layer.

        For each base weight W with a matching LoRA pair:
            W_merged = W + (lora_B @ lora_A) * scaling

        All matrix multiplications are performed in float32 for numerical
        stability, then cast back to the original weight dtype.
        """
        pairs = _extract_lora_pairs(self.lora_state_dict, ctx.layer_index)

        if not pairs:
            logger.debug(
                "No LoRA pairs found for layer %d.", ctx.layer_index
            )
            return state_dict

        merged_count = 0

        for module_path, (lora_a, lora_b) in pairs.items():
            if not self._should_merge_module(module_path):
                logger.debug(
                    "Skipping module %s (not in target_modules).", module_path
                )
                continue

            base_key = _find_base_key(state_dict, module_path)
            if base_key is None:
                logger.warning(
                    "Could not find base weight for module %s in layer %d "
                    "state dict. Available keys: %s",
                    module_path, ctx.layer_index, list(state_dict.keys()),
                )
                continue

            base_weight = state_dict[base_key]
            original_dtype = base_weight.dtype

            # Compute LoRA delta in float32
            # lora_A shape: [rank, in_features]
            # lora_B shape: [out_features, rank]
            # delta = lora_B @ lora_A -> [out_features, in_features]
            lora_a_f32 = lora_a.to(torch.float32)
            lora_b_f32 = lora_b.to(torch.float32)
            delta = lora_b_f32 @ lora_a_f32

            # Validate shapes
            if delta.shape != base_weight.shape:
                logger.error(
                    "Shape mismatch for %s: base=%s, delta=%s (lora_A=%s, lora_B=%s). "
                    "Skipping merge.",
                    base_key, base_weight.shape, delta.shape,
                    lora_a.shape, lora_b.shape,
                )
                continue

            # Merge: W_new = W_orig + delta * scaling
            base_f32 = base_weight.to(torch.float32)
            merged = base_f32 + delta * self.scaling

            state_dict[base_key] = merged.to(original_dtype)
            merged_count += 1

            logger.debug(
                "Merged LoRA for %s.%s: rank=%d, scaling=%.4f, "
                "delta_norm=%.6f, base_norm=%.6f",
                ctx.layer_name, module_path,
                lora_a.shape[0],
                self.scaling,
                torch.linalg.vector_norm(delta).item(),
                torch.linalg.vector_norm(base_f32).item(),
            )

        self._tensors_merged += merged_count
        self._layers_processed += 1

        logger.info(
            "Merged %d LoRA adapters into layer %d (%s).",
            merged_count, ctx.layer_index, ctx.layer_type,
        )

        return state_dict

    def validate(
        self,
        original: dict[str, Tensor],
        modified: dict[str, Tensor],
        ctx: LayerContext,
    ) -> bool:
        """Validate merge produced finite values and reasonable magnitudes."""
        if not super().validate(original, modified, ctx):
            return False

        # Verify merged weights are not drastically different
        for key in modified:
            if key not in original:
                continue

            orig_norm = torch.linalg.vector_norm(
                original[key].to(torch.float32)
            ).item()
            mod_norm = torch.linalg.vector_norm(
                modified[key].to(torch.float32)
            ).item()

            if orig_norm > 0:
                ratio = mod_norm / orig_norm
                # LoRA merges should not change norms by more than 100%
                if ratio > 2.0 or ratio < 0.01:
                    logger.error(
                        "LoRA merge changed norm of %s by factor %.3f "
                        "(original: %.3f, modified: %.3f). This may indicate "
                        "incorrect scaling or corrupted adapter weights.",
                        key, ratio, orig_norm, mod_norm,
                    )
                    return False

        return True

    def estimate_memory(self, ctx: LayerContext) -> int:
        """Estimate peak memory: need float32 copies for matmul."""
        pairs = _extract_lora_pairs(self.lora_state_dict, ctx.layer_index)
        peak = 0
        for module_path, (lora_a, lora_b) in pairs.items():
            # Memory for: lora_a_f32 + lora_b_f32 + delta_f32 + base_f32
            a_bytes = lora_a.numel() * 4
            b_bytes = lora_b.numel() * 4
            delta_bytes = lora_b.shape[0] * lora_a.shape[1] * 4
            base_bytes = delta_bytes  # Same shape as delta
            total = a_bytes + b_bytes + delta_bytes + base_bytes
            peak = max(peak, total)
        return peak

    @property
    def stats(self) -> dict:
        """Return summary statistics from the LoRA merge run."""
        return {
            "tensors_merged": self._tensors_merged,
            "layers_processed": self._layers_processed,
            "scaling": self.scaling,
            "lora_alpha": self.lora_alpha,
            "lora_rank": self.lora_rank,
            "layers_with_adapters": len(self._layer_indices_with_lora),
            "target_modules": self.target_modules,
        }
