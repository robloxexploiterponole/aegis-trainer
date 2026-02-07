"""
Weight inspection — collects per-tensor statistics without modifying weights.

Non-destructive analysis operation that gathers detailed statistics for every
tensor in each layer's state dict. Useful for pre/post modification auditing,
debugging weight distributions, identifying anomalous layers, and verifying
quantization quality.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor

from ..layer_context import LayerContext
from .base import LayerOperation

logger = logging.getLogger(__name__)


@dataclass
class TensorStats:
    """Statistics for a single tensor."""

    key: str
    shape: tuple[int, ...]
    dtype: str
    numel: int
    mean: float
    std: float
    min_val: float
    max_val: float
    abs_mean: float
    abs_max: float
    sparsity: float  # Fraction of exact zeros
    near_zero_fraction: float  # Fraction of values < 1e-6 in absolute value
    has_nan: bool
    has_inf: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for serialization."""
        return {
            "key": self.key,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "numel": self.numel,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_val,
            "max": self.max_val,
            "abs_mean": self.abs_mean,
            "abs_max": self.abs_max,
            "sparsity": self.sparsity,
            "near_zero_fraction": self.near_zero_fraction,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
        }


@dataclass
class LayerStats:
    """Aggregated statistics for a single layer."""

    layer_index: int
    layer_type: str
    layer_name: str
    tensor_count: int
    total_params: int
    total_bytes: int
    tensors: dict[str, TensorStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to plain dict for serialization."""
        return {
            "layer_index": self.layer_index,
            "layer_type": self.layer_type,
            "layer_name": self.layer_name,
            "tensor_count": self.tensor_count,
            "total_params": self.total_params,
            "total_bytes": self.total_bytes,
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
        }


def _dtype_size(dtype: torch.dtype) -> int:
    """Return byte size per element for a given dtype."""
    sizes = {
        torch.float64: 8,
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return sizes.get(dtype, 4)  # Default to 4 bytes for unknown dtypes


class WeightInspectOp(LayerOperation):
    """Weight inspection operation — collects statistics without modification.

    Gathers per-tensor statistics including mean, std, min, max, sparsity,
    dtype, and shape. Results are stored internally and can be retrieved
    via get_results() or printed with print_report().

    Args:
        collect_stats: Whether to actually collect statistics. If False,
            the operation is a pass-through. Defaults to True.
        near_zero_threshold: Threshold for counting near-zero values.
            Defaults to 1e-6.
    """

    name: str = "weight_inspect"

    def __init__(
        self,
        collect_stats: bool = True,
        near_zero_threshold: float = 1e-6,
    ) -> None:
        self.collect_stats = collect_stats
        self.near_zero_threshold = near_zero_threshold
        self._results: dict[int, LayerStats] = {}

    def should_apply(self, ctx: LayerContext) -> bool:
        """Apply to all layers — inspection is non-destructive."""
        return True

    def _compute_tensor_stats(self, key: str, tensor: Tensor) -> TensorStats:
        """Compute statistics for a single tensor.

        Casts to float32 for accurate statistics regardless of storage dtype.
        """
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()

        # Cast to float32 for accurate statistics
        t = tensor.to(torch.float32)

        # Filter out NaN/Inf for stats computation
        finite_mask = torch.isfinite(t)
        if finite_mask.any():
            finite_vals = t[finite_mask]
            mean_val = finite_vals.mean().item()
            std_val = finite_vals.std().item() if finite_vals.numel() > 1 else 0.0
            min_val = finite_vals.min().item()
            max_val = finite_vals.max().item()
            abs_vals = finite_vals.abs()
            abs_mean = abs_vals.mean().item()
            abs_max = abs_vals.max().item()
        else:
            mean_val = float("nan")
            std_val = float("nan")
            min_val = float("nan")
            max_val = float("nan")
            abs_mean = float("nan")
            abs_max = float("nan")

        numel = tensor.numel()
        sparsity = (tensor == 0).sum().item() / numel if numel > 0 else 0.0
        near_zero = (
            (tensor.abs() < self.near_zero_threshold).sum().item() / numel
            if numel > 0
            else 0.0
        )

        return TensorStats(
            key=key,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            numel=numel,
            mean=mean_val,
            std=std_val,
            min_val=min_val,
            max_val=max_val,
            abs_mean=abs_mean,
            abs_max=abs_max,
            sparsity=sparsity,
            near_zero_fraction=near_zero,
            has_nan=has_nan,
            has_inf=has_inf,
        )

    def apply(
        self,
        state_dict: dict[str, Tensor],
        ctx: LayerContext,
    ) -> dict[str, Tensor]:
        """Collect per-tensor stats for this layer.

        Returns state_dict UNMODIFIED. This operation is purely observational.
        """
        if not self.collect_stats:
            return state_dict

        tensor_stats: dict[str, TensorStats] = {}
        total_params = 0
        total_bytes = 0

        for key, tensor in state_dict.items():
            stats = self._compute_tensor_stats(key, tensor)
            tensor_stats[key] = stats
            total_params += tensor.numel()
            total_bytes += tensor.numel() * _dtype_size(tensor.dtype)

            if stats.has_nan:
                logger.warning(
                    "NaN detected in %s layer %d", key, ctx.layer_index
                )
            if stats.has_inf:
                logger.warning(
                    "Inf detected in %s layer %d", key, ctx.layer_index
                )

        layer_stats = LayerStats(
            layer_index=ctx.layer_index,
            layer_type=ctx.layer_type,
            layer_name=ctx.layer_name,
            tensor_count=len(tensor_stats),
            total_params=total_params,
            total_bytes=total_bytes,
            tensors=tensor_stats,
        )

        self._results[ctx.layer_index] = layer_stats

        logger.info(
            "Inspected layer %d (%s): %d tensors, %d params, %.2f MB",
            ctx.layer_index,
            ctx.layer_type,
            len(tensor_stats),
            total_params,
            total_bytes / (1024 * 1024),
        )

        return state_dict

    def get_results(self) -> dict[int, LayerStats]:
        """Return all collected layer statistics.

        Returns:
            Dict mapping layer index to LayerStats dataclass.
        """
        return self._results

    def get_results_dict(self) -> dict[int, dict]:
        """Return all results as plain dicts for serialization."""
        return {idx: stats.to_dict() for idx, stats in self._results.items()}

    def get_summary(self) -> dict[str, Any]:
        """Return a high-level summary across all inspected layers."""
        if not self._results:
            return {"layers_inspected": 0}

        total_params = sum(s.total_params for s in self._results.values())
        total_bytes = sum(s.total_bytes for s in self._results.values())
        total_tensors = sum(s.tensor_count for s in self._results.values())

        # Count anomalies
        nan_count = 0
        inf_count = 0
        high_sparsity_count = 0

        for layer_stats in self._results.values():
            for ts in layer_stats.tensors.values():
                if ts.has_nan:
                    nan_count += 1
                if ts.has_inf:
                    inf_count += 1
                if ts.sparsity > 0.5:
                    high_sparsity_count += 1

        return {
            "layers_inspected": len(self._results),
            "total_tensors": total_tensors,
            "total_params": total_params,
            "total_bytes": total_bytes,
            "total_gb": total_bytes / (1024**3),
            "tensors_with_nan": nan_count,
            "tensors_with_inf": inf_count,
            "tensors_high_sparsity": high_sparsity_count,
        }

    def print_report(self) -> None:
        """Print a human-readable report of all collected statistics."""
        summary = self.get_summary()
        print("=" * 80)
        print("AEGIS Weight Inspection Report")
        print("=" * 80)
        print(f"Layers inspected: {summary['layers_inspected']}")
        print(f"Total tensors:    {summary.get('total_tensors', 0)}")
        print(f"Total parameters: {summary.get('total_params', 0):,}")
        print(f"Total size:       {summary.get('total_gb', 0):.2f} GB")
        print(f"Tensors with NaN: {summary.get('tensors_with_nan', 0)}")
        print(f"Tensors with Inf: {summary.get('tensors_with_inf', 0)}")
        print(f"High sparsity:    {summary.get('tensors_high_sparsity', 0)}")
        print("-" * 80)

        for layer_idx in sorted(self._results.keys()):
            layer = self._results[layer_idx]
            print(
                f"\nLayer {layer_idx:3d} ({layer.layer_type:20s}) "
                f"— {layer.tensor_count} tensors, "
                f"{layer.total_params:,} params, "
                f"{layer.total_bytes / (1024 * 1024):.1f} MB"
            )

            for key in sorted(layer.tensors.keys()):
                ts = layer.tensors[key]
                flags = []
                if ts.has_nan:
                    flags.append("NaN!")
                if ts.has_inf:
                    flags.append("Inf!")
                if ts.sparsity > 0.5:
                    flags.append(f"sparse({ts.sparsity:.1%})")

                flag_str = f" [{', '.join(flags)}]" if flags else ""
                print(
                    f"  {key:60s}  "
                    f"shape={list(ts.shape)!s:20s}  "
                    f"dtype={ts.dtype:10s}  "
                    f"mean={ts.mean:+.4e}  "
                    f"std={ts.std:.4e}  "
                    f"range=[{ts.min_val:+.4e}, {ts.max_val:+.4e}]"
                    f"{flag_str}"
                )

        print("\n" + "=" * 80)
