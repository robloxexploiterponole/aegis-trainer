"""
Layer-level quantization/compression — casts all tensors to a target dtype.

Provides simple dtype conversion for model weights during the layer-streaming
pipeline. Useful for converting fp32 models to fp16/bf16, or for applying
uniform dtype to mixed-precision checkpoints.

For more advanced quantization (GPTQ, AWQ, etc.), use dedicated quantization
tools. This operation handles straightforward dtype casting with optional
validation of precision loss.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch import Tensor

from ..layer_context import LayerContext
from .base import LayerOperation

logger = logging.getLogger(__name__)

# Supported target dtypes for quantization
_SUPPORTED_DTYPES = {
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
}

# Dtype size in bytes for memory estimation
_DTYPE_BYTES = {
    torch.float64: 8,
    torch.float32: 4,
    torch.float16: 2,
    torch.bfloat16: 2,
}


class QuantizeOp(LayerOperation):
    """Layer-level quantization operation — casts all tensors to target dtype.

    Simple but effective dtype conversion for uniform model weight compression.
    Casts every tensor in the layer's state dict to the specified dtype.

    Args:
        target_dtype: Target dtype for all weight tensors. Defaults to
            torch.float16.
        skip_non_float: If True, skip tensors that are not floating-point
            (e.g. integer indices, boolean masks). Defaults to True.
        max_abs_error_threshold: Optional threshold for maximum absolute error
            when casting. If set, validation will fail if any element's
            casting error exceeds this threshold. Default None (no check).
    """

    name: str = "quantize"

    def __init__(
        self,
        target_dtype: torch.dtype = torch.float16,
        skip_non_float: bool = True,
        max_abs_error_threshold: float | None = None,
    ) -> None:
        if target_dtype not in _SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported target_dtype: {target_dtype}. "
                f"Supported: {_SUPPORTED_DTYPES}"
            )

        self.target_dtype = target_dtype
        self.skip_non_float = skip_non_float
        self.max_abs_error_threshold = max_abs_error_threshold

        # Stats
        self._tensors_cast = 0
        self._tensors_skipped = 0
        self._layers_processed = 0
        self._total_bytes_saved = 0

    def should_apply(self, ctx: LayerContext) -> bool:
        """Apply to all layers — quantization is universal."""
        return True

    def apply(
        self,
        state_dict: dict[str, Tensor],
        ctx: LayerContext,
    ) -> dict[str, Tensor]:
        """Cast all tensors to target dtype.

        Non-float tensors (integer indices, boolean masks) are optionally
        skipped to preserve their exact values. Tensors already at the
        target dtype are returned as-is.

        Args:
            state_dict: Layer state dict containing weight tensors.
            ctx: LayerContext with layer metadata.

        Returns:
            State dict with all eligible tensors cast to target dtype.
        """
        cast_count = 0
        skip_count = 0
        bytes_saved = 0

        for key in list(state_dict.keys()):
            tensor = state_dict[key]

            # Skip non-float tensors if configured
            if self.skip_non_float and not tensor.is_floating_point():
                skip_count += 1
                logger.debug(
                    "Skipping non-float tensor %s (dtype=%s)", key, tensor.dtype
                )
                continue

            # Skip if already at target dtype
            if tensor.dtype == self.target_dtype:
                continue

            original_dtype = tensor.dtype
            original_bytes = tensor.numel() * _DTYPE_BYTES.get(original_dtype, 4)

            # Cast to target dtype
            state_dict[key] = tensor.to(self.target_dtype)

            new_bytes = tensor.numel() * _DTYPE_BYTES.get(self.target_dtype, 4)
            bytes_saved += original_bytes - new_bytes
            cast_count += 1

            logger.debug(
                "Cast %s: %s -> %s (saved %d bytes)",
                key, original_dtype, self.target_dtype,
                original_bytes - new_bytes,
            )

        self._tensors_cast += cast_count
        self._tensors_skipped += skip_count
        self._layers_processed += 1
        self._total_bytes_saved += bytes_saved

        logger.info(
            "Quantized layer %d (%s): %d tensors cast to %s, "
            "%d skipped, %.2f MB saved",
            ctx.layer_index,
            ctx.layer_type,
            cast_count,
            self.target_dtype,
            skip_count,
            bytes_saved / (1024 * 1024),
        )

        return state_dict

    def validate(
        self,
        original: dict[str, Tensor],
        modified: dict[str, Tensor],
        ctx: LayerContext,
    ) -> bool:
        """Validate that quantization did not introduce excessive errors.

        Checks:
          1. No NaN or Inf values introduced.
          2. All eligible tensors are at the target dtype.
          3. Optional: maximum absolute error within threshold.
        """
        if not super().validate(original, modified, ctx):
            return False

        # Verify dtypes
        for key, tensor in modified.items():
            if self.skip_non_float and not tensor.is_floating_point():
                continue
            if tensor.dtype != self.target_dtype:
                logger.error(
                    "Tensor %s has dtype %s, expected %s after quantization.",
                    key, tensor.dtype, self.target_dtype,
                )
                return False

        # Check precision loss if threshold is set
        if self.max_abs_error_threshold is not None:
            for key in modified:
                if key not in original:
                    continue
                orig = original[key]
                mod = modified[key]

                if not orig.is_floating_point():
                    continue

                # Compare in float32 to measure actual error
                orig_f32 = orig.to(torch.float32)
                mod_f32 = mod.to(torch.float32)

                max_error = (orig_f32 - mod_f32).abs().max().item()
                if max_error > self.max_abs_error_threshold:
                    logger.error(
                        "Quantization error for %s exceeds threshold: "
                        "max_abs_error=%.6e > threshold=%.6e",
                        key, max_error, self.max_abs_error_threshold,
                    )
                    return False

        return True

    def estimate_memory(self, ctx: LayerContext) -> int:
        """Estimate peak memory: torch.to() may temporarily hold both copies."""
        # During casting, PyTorch may hold the original and new tensor simultaneously.
        # Estimate based on the larger dtype (float32) for a typical layer.
        # Rough: 512 experts * avg tensor size
        # A single expert has ~3 tensors of ~2048 * 1024 * 4 bytes each
        return ctx.hidden_size * 1024 * 4 * 2  # One tensor in both dtypes

    @property
    def stats(self) -> dict:
        """Return summary statistics from the quantization run."""
        return {
            "tensors_cast": self._tensors_cast,
            "tensors_skipped": self._tensors_skipped,
            "layers_processed": self._layers_processed,
            "total_bytes_saved": self._total_bytes_saved,
            "total_mb_saved": self._total_bytes_saved / (1024 * 1024),
            "target_dtype": str(self.target_dtype),
        }
