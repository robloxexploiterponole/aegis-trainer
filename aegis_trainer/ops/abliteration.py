"""
Directional ablation — removes "refusal directions" from weight matrices.

Adapted from Heretic's LoRA-based ablation algorithm for direct weight
modification. Projects out refusal-correlated directions from o_proj and
down_proj weight matrices across all 48 transformer layers, targeting all
512 MoE experts (not just the 10 active ones).

Algorithm:
    For each target weight matrix W:
        direction = refusal_directions[direction_index]  # [hidden_size]
        projection = (W @ d_col) @ d_row   where d_col = d[:, None], d_row = d[None, :]
        W_modified = W - projection * component_weight

    This removes the component of W that maps into the refusal direction.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ..layer_context import LayerContext
from .base import LayerOperation

logger = logging.getLogger(__name__)

# Default modules targeted for abliteration
_DEFAULT_TARGET_MODULES = ["o_proj", "down_proj"]

# Regex for matching expert weight keys:
# model.layers.{i}.mlp.experts.{N}.{module}.weight
_EXPERT_KEY_PATTERN = re.compile(
    r"^(?:.*\.)?mlp\.experts\.(\d+)\.(\w+)\.weight$"
)
# Regex for shared expert:
# model.layers.{i}.mlp.shared_expert.{module}.weight
_SHARED_EXPERT_KEY_PATTERN = re.compile(
    r"^(?:.*\.)?mlp\.shared_expert\.(\w+)\.weight$"
)
# Regex for attention output projection:
# model.layers.{i}.self_attn.o_proj.weight
_ATTN_KEY_PATTERN = re.compile(
    r"^(?:.*\.)?self_attn\.(\w+)\.weight$"
)


class AbliterationOp(LayerOperation):
    """Directional ablation operation — projects refusal directions out of weights.

    Works on all 48 transformer layers (both DeltaNet linear_attention and
    full_attention). Targets o_proj in self-attention and down_proj in all 512
    MoE experts plus the shared expert.

    Args:
        refusal_directions: Tensor of shape [num_directions, hidden_size] containing
            the refusal direction vectors (typically computed from residual stream
            analysis of harmful vs harmless prompts).
        direction_index: Which direction vector to use from refusal_directions.
            Defaults to 0 (the primary refusal direction).
        component_weights: Optional per-layer weight scaling. If provided, must
            have length >= total_layers. Scales the projection removal per layer.
            Values of 0.0 skip that layer; 1.0 removes the full projection.
        target_modules: Which module types to target. Defaults to
            ["o_proj", "down_proj"].
        row_normalization: If True, preserve row norms of the weight matrix
            after ablation (norm-preserving abliteration).
    """

    name: str = "abliterate"

    def __init__(
        self,
        refusal_directions: Tensor,
        direction_index: int = 0,
        component_weights: list[float] | None = None,
        target_modules: list[str] | None = None,
        row_normalization: bool = False,
    ) -> None:
        if refusal_directions.ndim == 1:
            # Single direction vector — wrap to [1, hidden_size]
            refusal_directions = refusal_directions.unsqueeze(0)

        if refusal_directions.ndim != 2:
            raise ValueError(
                f"refusal_directions must be 2D [num_directions, hidden_size], "
                f"got shape {refusal_directions.shape}"
            )

        if direction_index < 0 or direction_index >= refusal_directions.shape[0]:
            raise ValueError(
                f"direction_index {direction_index} out of range for "
                f"{refusal_directions.shape[0]} directions"
            )

        self.refusal_directions = refusal_directions.to(torch.float32)
        self.direction_index = direction_index
        self.component_weights = component_weights
        self.target_modules = target_modules or _DEFAULT_TARGET_MODULES
        self.row_normalization = row_normalization

        # Pre-normalize the selected direction for numerical stability
        self._direction = F.normalize(
            self.refusal_directions[self.direction_index], p=2, dim=0
        )

        # Stats tracking
        self._tensors_modified = 0
        self._layers_processed = 0

    def should_apply(self, ctx: LayerContext) -> bool:
        """Apply to ALL 48 transformer layers (both DeltaNet and full_attention).

        Skip embed_tokens, norm, lm_head — those are not transformer layers
        and are handled outside the layer loop.
        """
        return ctx.layer_type in ("linear_attention", "full_attention")

    def _get_layer_weight(self, ctx: LayerContext) -> float:
        """Get the ablation weight for this specific layer.

        Returns a scaling factor in [0, 1] that controls how aggressively
        the refusal direction is projected out of this layer's weights.
        """
        if self.component_weights is not None:
            if ctx.layer_index < len(self.component_weights):
                return self.component_weights[ctx.layer_index]
            return 0.0
        return 1.0

    def _should_target_key(self, key: str) -> bool:
        """Check if a state_dict key matches one of our target modules."""
        for target in self.target_modules:
            if f".{target}.weight" in key or key.endswith(f"{target}.weight"):
                return True
        return False

    def _abliterate_tensor(
        self,
        weight: Tensor,
        direction: Tensor,
        layer_weight: float,
    ) -> Tensor:
        """Remove the refusal direction component from a weight matrix.

        Args:
            weight: Weight tensor of shape [out_features, in_features].
            direction: Normalized refusal direction of shape [hidden_size].
            layer_weight: Scaling factor for the projection removal.

        Returns:
            Modified weight tensor with refusal direction projected out.
        """
        original_dtype = weight.dtype
        original_device = weight.device

        # Cast to float32 for numerical stability
        W = weight.to(torch.float32)

        # Move direction to the same device
        d = direction.to(W.device)

        # Determine which dimension the direction applies to.
        # For o_proj: shape is [hidden_size, num_heads * head_dim]
        #   direction applies along dim=0 (output dimension)
        # For down_proj: shape is [hidden_size, intermediate_size]
        #   direction applies along dim=0 (output dimension)
        #
        # Both cases: direction is [hidden_size] and maps to the output dimension.
        out_features, in_features = W.shape

        if out_features == d.shape[0]:
            # Standard case: direction aligns with output features
            # projection = (W^T @ d) @ d^T  = outer product projection
            # Equivalently: W_modified = W - d[:, None] @ (d[None, :] @ W)
            # This removes the component of each column of W that lies along d.

            if self.row_normalization:
                # Preserve row norms after ablation
                row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)

            # Compute projection: for each row w_i of W, remove (w_i . d) * d component
            # But since we want to remove from columns: projection = d * (d^T @ W)
            projection = d.unsqueeze(-1) @ (d.unsqueeze(0) @ W).unsqueeze(0)
            # projection shape: [1, out_features, in_features] -> squeeze
            projection = projection.squeeze(0)

            W = W - layer_weight * projection

            if self.row_normalization:
                # Re-normalize rows to preserve original magnitudes
                new_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
                # Avoid division by zero
                scale = row_norms / (new_norms + 1e-8)
                W = W * scale

        elif in_features == d.shape[0]:
            # Transposed case: direction aligns with input features
            # Remove the component of each row that projects onto direction

            if self.row_normalization:
                row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)

            # For each row w_i: w_i_new = w_i - (w_i . d) * d
            projections = (W @ d.unsqueeze(-1))  # [out, 1]
            projection = projections @ d.unsqueeze(0)  # [out, in]

            W = W - layer_weight * projection

            if self.row_normalization:
                new_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True)
                scale = row_norms / (new_norms + 1e-8)
                W = W * scale
        else:
            logger.warning(
                "Direction size %d does not match weight dimensions (%d, %d), "
                "skipping ablation for this tensor.",
                d.shape[0], out_features, in_features,
            )
            return weight

        # Cast back to original dtype
        return W.to(original_dtype)

    def apply(
        self,
        state_dict: dict[str, Tensor],
        ctx: LayerContext,
    ) -> dict[str, Tensor]:
        """Apply directional ablation to all target weight matrices in this layer.

        Processes:
          - self_attn.o_proj.weight
          - mlp.experts.{0-511}.down_proj.weight  (ALL 512 experts)
          - mlp.shared_expert.down_proj.weight

        All computations are done in float32 for numerical stability, then
        cast back to original dtype.
        """
        layer_weight = self._get_layer_weight(ctx)
        if layer_weight == 0.0:
            logger.debug(
                "Skipping ablation for layer %d (weight=0.0)", ctx.layer_index
            )
            return state_dict

        direction = self._direction
        modified_count = 0

        for key in list(state_dict.keys()):
            if not self._should_target_key(key):
                continue

            tensor = state_dict[key]
            if tensor.ndim != 2:
                logger.debug(
                    "Skipping non-matrix tensor %s (shape %s)", key, tensor.shape
                )
                continue

            state_dict[key] = self._abliterate_tensor(
                tensor, direction, layer_weight
            )
            modified_count += 1

        self._tensors_modified += modified_count
        self._layers_processed += 1

        logger.info(
            "Abliterated layer %d (%s): %d tensors modified (weight=%.3f)",
            ctx.layer_index,
            ctx.layer_type,
            modified_count,
            layer_weight,
        )

        return state_dict

    def validate(
        self,
        original: dict[str, Tensor],
        modified: dict[str, Tensor],
        ctx: LayerContext,
    ) -> bool:
        """Validate that ablation did not introduce NaN/Inf values and that
        weight magnitudes did not change drastically."""
        # Run base validation for NaN/Inf
        if not super().validate(original, modified, ctx):
            return False

        # Check that modification magnitudes are reasonable
        for key in modified:
            if key not in original:
                continue
            if not self._should_target_key(key):
                continue

            orig_norm = torch.linalg.vector_norm(
                original[key].to(torch.float32)
            ).item()
            mod_norm = torch.linalg.vector_norm(
                modified[key].to(torch.float32)
            ).item()

            if orig_norm > 0:
                ratio = mod_norm / orig_norm
                # Weight norms should not change by more than 50%
                if ratio < 0.5 or ratio > 1.5:
                    logger.warning(
                        "Ablation changed norm of %s by factor %.3f "
                        "(original: %.3f, modified: %.3f)",
                        key, ratio, orig_norm, mod_norm,
                    )
                    # Warning only — don't fail validation for moderate changes

        return True

    def estimate_memory(self, ctx: LayerContext) -> int:
        """Estimate peak memory: need float32 copies of target tensors."""
        # Rough estimate: o_proj + 512 expert down_proj + shared expert down_proj
        # o_proj: [hidden_size, num_heads * head_dim] = [2048, 4096]
        o_proj_bytes = ctx.hidden_size * ctx.num_attention_heads * ctx.head_dim * 4
        # down_proj per expert: [hidden_size, intermediate_size]
        # Intermediate size for Qwen3-Next ~ 1024 per expert (small experts)
        # We process one tensor at a time, so peak is max(single_tensor) * 2 (orig + float32)
        return o_proj_bytes * 2  # Conservative: largest single tensor * 2

    @property
    def stats(self) -> dict:
        """Return summary statistics from the ablation run."""
        return {
            "tensors_modified": self._tensors_modified,
            "layers_processed": self._layers_processed,
            "direction_index": self.direction_index,
            "row_normalization": self.row_normalization,
            "target_modules": self.target_modules,
        }
