"""
LongRoPE — extends context window by modifying RoPE scaling factors.

Applies rescale factors from LongRoPE evolutionary search to modify inverse
frequency buffers in full_attention layers. Only targets the 12 full_attention
layers out of 48 total (DeltaNet layers use linear attention with delta rules
and have no RoPE).

For Qwen3-Next, RoPE is computed dynamically from config, so the primary
mechanism is either:
  1. Modifying inv_freq buffers if they exist in the state_dict.
  2. Patching config.json with new RoPE scaling parameters.

Reference: Microsoft LongRoPE (arXiv:2402.13753)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

from ..layer_context import LayerContext
from .base import LayerOperation

logger = logging.getLogger(__name__)


def _calc_mscale_su(scale: float, original_max_position_embeddings: int) -> float:
    """Compute magnitude scaling factor using the SU (scale-up) policy.

    This corrects for the attention temperature change that occurs when
    extending the context window.
    """
    if scale <= 1.0:
        return 1.0
    return math.sqrt(
        1.0 + math.log(scale) / math.log(original_max_position_embeddings)
    )


def _calc_mscale_yarn(scale: float) -> float:
    """Compute magnitude scaling factor using YaRN policy."""
    if scale <= 1.0:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class LongRoPEOp(LayerOperation):
    """LongRoPE context window extension operation.

    Modifies RoPE inverse frequency buffers to extend the model's maximum
    context length. Only applies to full_attention layers (12 out of 48
    in Qwen3-Next) since DeltaNet layers use linear attention without RoPE.

    Args:
        rescale_factors: Tensor of rescale factors from LongRoPE evolution search.
            Shape: [head_dim // 2] — one factor per frequency dimension pair.
        target_max_position_embeddings: Target maximum context length (e.g. 524288
            for 512K context).
        original_max_position_embeddings: Original maximum context length before
            extension (e.g. 32768 for 32K).
        rope_base: Base frequency for RoPE computation. Default 10000.0.
        magnitude_scaling_policy: How to scale attention magnitudes after extension.
            One of "su" (default), "yarn", or a fixed float string.
    """

    name: str = "longrope"

    def __init__(
        self,
        rescale_factors: Tensor,
        target_max_position_embeddings: int = 524288,
        original_max_position_embeddings: int = 32768,
        rope_base: float = 10000.0,
        magnitude_scaling_policy: str = "su",
    ) -> None:
        self.rescale_factors = rescale_factors.to(torch.float32)
        self.target_max_position_embeddings = target_max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_base = rope_base
        self.magnitude_scaling_policy = magnitude_scaling_policy

        # Compute scale ratio and magnitude scaling
        self.scale_ratio = (
            target_max_position_embeddings / original_max_position_embeddings
        )

        if magnitude_scaling_policy == "su":
            self.mscale = _calc_mscale_su(
                self.scale_ratio, original_max_position_embeddings
            )
        elif magnitude_scaling_policy == "yarn":
            self.mscale = _calc_mscale_yarn(self.scale_ratio)
        else:
            try:
                self.mscale = float(magnitude_scaling_policy)
            except ValueError:
                raise ValueError(
                    f"Unknown magnitude_scaling_policy: {magnitude_scaling_policy!r}. "
                    f"Use 'su', 'yarn', or a numeric string."
                )

        # Stats
        self._layers_modified = 0
        self._inv_freq_modified = 0
        self._config_patched = False

    def should_apply(self, ctx: LayerContext) -> bool:
        """ONLY apply to full_attention layers (12 of 48).

        DeltaNet layers use linear attention with delta update rules and
        do not use rotary position embeddings.
        """
        return ctx.is_rope_enabled

    def _compute_inv_freq(self, head_dim: int, device: torch.device) -> Tensor:
        """Compute rescaled inverse frequencies for LongRoPE.

        Uses the same formula as LongRoPEScaledRotaryEmbedding._calc_inv_freq:
            inv_freq = 1.0 / (rescale_factors * (base ^ (2i / dim)))

        Args:
            head_dim: Per-head dimension (e.g. 256 for Qwen3-Next).
            device: Target device for the tensor.

        Returns:
            Tensor of shape [head_dim // 2] with rescaled inverse frequencies.
        """
        dim = head_dim
        factors = self.rescale_factors.to(device)

        # Validate dimension alignment
        expected_size = dim // 2
        if factors.shape[0] != expected_size:
            raise ValueError(
                f"rescale_factors has {factors.shape[0]} elements but head_dim={dim} "
                f"requires {expected_size} elements (head_dim // 2)."
            )

        exponent = torch.arange(
            0, dim, 2, dtype=torch.float32, device=device
        ) / dim
        base_freq = self.rope_base ** exponent
        inv_freq = 1.0 / (factors * base_freq)

        return inv_freq

    def apply(
        self,
        state_dict: dict[str, Tensor],
        ctx: LayerContext,
    ) -> dict[str, Tensor]:
        """Modify RoPE frequencies in the layer's state dict.

        If inv_freq buffer exists in the state dict, apply rescale_factors
        to generate new inverse frequencies. For Qwen3-Next where RoPE is
        typically computed dynamically from config, this handles the case
        where inv_freq is materialized as a buffer.

        The primary mechanism for extending context in dynamic-RoPE models
        is patching config.json via the patch_config() static method.
        """
        inv_freq_keys = [
            k for k in state_dict.keys()
            if "inv_freq" in k
        ]

        if inv_freq_keys:
            # Compute rescaled inverse frequencies
            new_inv_freq = self._compute_inv_freq(
                ctx.head_dim, torch.device("cpu")
            )

            for key in inv_freq_keys:
                original = state_dict[key]
                original_dtype = original.dtype

                if original.shape != new_inv_freq.shape:
                    logger.warning(
                        "inv_freq shape mismatch in %s: expected %s, got %s. "
                        "Attempting to reshape.",
                        key, new_inv_freq.shape, original.shape,
                    )
                    # Try to match shapes
                    if original.numel() == new_inv_freq.numel():
                        new_inv_freq = new_inv_freq.reshape(original.shape)
                    else:
                        logger.error(
                            "Cannot reconcile inv_freq shapes for %s. Skipping.", key
                        )
                        continue

                state_dict[key] = new_inv_freq.to(original_dtype)
                self._inv_freq_modified += 1
                logger.info(
                    "Modified inv_freq %s for layer %d (scale: %.1fx, mscale: %.4f)",
                    key, ctx.layer_index, self.scale_ratio, self.mscale,
                )
        else:
            logger.debug(
                "No inv_freq buffer found in layer %d state dict. "
                "RoPE extension requires config.json patching for this model.",
                ctx.layer_index,
            )

        self._layers_modified += 1
        return state_dict

    def validate(
        self,
        original: dict[str, Tensor],
        modified: dict[str, Tensor],
        ctx: LayerContext,
    ) -> bool:
        """Validate inv_freq values are finite and positive."""
        if not super().validate(original, modified, ctx):
            return False

        for key, tensor in modified.items():
            if "inv_freq" in key:
                # Inverse frequencies must be positive and finite
                if (tensor <= 0).any():
                    logger.error(
                        "Validation failed: non-positive inv_freq values in %s", key
                    )
                    return False
        return True

    @staticmethod
    def patch_config(
        config_path: Path,
        new_max_position: int,
        rope_scaling: dict[str, Any],
    ) -> None:
        """Patch the model's config.json with new RoPE settings.

        This is the primary mechanism for extending context length in models
        that compute RoPE dynamically from config (like Qwen3-Next).

        Args:
            config_path: Path to the model's config.json file.
            new_max_position: New max_position_embeddings value.
            rope_scaling: Dict with RoPE scaling configuration. Example:
                {
                    "type": "longrope",
                    "short_factor": [...],  # rescale factors for short context
                    "long_factor": [...],   # rescale factors for long context
                    "original_max_position_embeddings": 32768,
                    "short_mscale": 1.0,
                    "long_mscale": 1.05,
                }
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # Backup original values
        original_max_pos = config.get("max_position_embeddings", "N/A")

        # Update config
        config["max_position_embeddings"] = new_max_position
        config["rope_scaling"] = rope_scaling

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(
            "Patched config.json: max_position_embeddings %s -> %d, "
            "added rope_scaling configuration.",
            original_max_pos,
            new_max_position,
        )

    @staticmethod
    def generate_rope_scaling_config(
        rescale_factors: Tensor,
        original_max_position_embeddings: int = 32768,
        target_max_position_embeddings: int = 524288,
        magnitude_scaling_policy: str = "su",
    ) -> dict[str, Any]:
        """Generate a rope_scaling config dict from rescale factors.

        Convenience method that formats LongRoPE parameters for config.json.

        Args:
            rescale_factors: Tensor of rescale factors from evolution search.
            original_max_position_embeddings: Original context length.
            target_max_position_embeddings: Target extended context length.
            magnitude_scaling_policy: Magnitude scaling policy ("su" or "yarn").

        Returns:
            Dict suitable for config.json["rope_scaling"].
        """
        scale = target_max_position_embeddings / original_max_position_embeddings

        if magnitude_scaling_policy == "su":
            mscale = _calc_mscale_su(scale, original_max_position_embeddings)
        elif magnitude_scaling_policy == "yarn":
            mscale = _calc_mscale_yarn(scale)
        else:
            mscale = float(magnitude_scaling_policy)

        factors_list = rescale_factors.to(torch.float32).tolist()

        return {
            "type": "longrope",
            "long_factor": factors_list,
            "short_factor": [1.0] * len(factors_list),
            "original_max_position_embeddings": original_max_position_embeddings,
            "long_mscale": mscale,
            "short_mscale": 1.0,
        }

    @property
    def stats(self) -> dict:
        """Return summary statistics from the LongRoPE run."""
        return {
            "layers_modified": self._layers_modified,
            "inv_freq_modified": self._inv_freq_modified,
            "config_patched": self._config_patched,
            "scale_ratio": self.scale_ratio,
            "mscale": self.mscale,
            "target_max_position": self.target_max_position_embeddings,
        }
