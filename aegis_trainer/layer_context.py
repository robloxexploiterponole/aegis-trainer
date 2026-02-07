"""
LayerContext — Frozen dataclass carrying per-layer metadata for AEGIS operations.

Carries all information an operation needs to process a single layer:
layer index, type (linear_attention vs full_attention), model architecture
parameters, checkpoint paths, device/dtype configuration, and MoE settings.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

try:
    from transformers import PretrainedConfig
except ImportError:
    PretrainedConfig = None


# Qwen3-Next architecture defaults
_QWEN3_NEXT_TOTAL_LAYERS = 48
_QWEN3_NEXT_NUM_EXPERTS = 512
_QWEN3_NEXT_NUM_ACTIVE_EXPERTS = 10
_QWEN3_NEXT_HIDDEN_SIZE = 2048
_QWEN3_NEXT_HEAD_DIM = 256
_QWEN3_NEXT_NUM_ATTENTION_HEADS = 16
_QWEN3_NEXT_NUM_KV_HEADS = 2

# Layer pattern: 3 linear_attention + 1 full_attention, repeating 12 times
_LAYER_PATTERN = ["linear_attention"] * 3 + ["full_attention"]
_LAYER_TYPES = _LAYER_PATTERN * 12  # 48 layers total


def _get_layer_type(index: int, config=None) -> str:
    """Determine layer type from config or default Qwen3-Next pattern.

    Args:
        index: Layer index (0-based).
        config: Optional HuggingFace PretrainedConfig with layer_types attribute.

    Returns:
        "linear_attention" or "full_attention".
    """
    if config is not None and hasattr(config, "layer_types") and config.layer_types:
        layer_types = config.layer_types
        if index < len(layer_types):
            return layer_types[index]
    # Fall back to default Qwen3-Next pattern
    if index < len(_LAYER_TYPES):
        return _LAYER_TYPES[index]
    # Beyond known range — use pattern cycling
    return _LAYER_PATTERN[index % len(_LAYER_PATTERN)]


@dataclass(frozen=True)
class LayerContext:
    """Immutable per-layer metadata passed to every AEGIS operation.

    Frozen to prevent accidental mutation during layer processing. All
    architecture-specific values default to Qwen3-Next settings but can
    be overridden via from_config() or direct construction.

    Attributes:
        layer_index: Zero-based layer index within the model.
        layer_name: Full layer prefix string (e.g. "model.layers.0.").
        layer_type: "linear_attention" (DeltaNet) or "full_attention" (RoPE/GQA).
        total_layers: Total transformer layers in the model.
        checkpoint_path: Path to the splitted_model directory.
        config: HuggingFace PretrainedConfig for the model.
        device: Target device for tensor operations.
        dtype: Target dtype for tensor operations.
        num_experts: Total MoE experts per layer.
        num_active_experts: Experts activated per token.
        hidden_size: Model hidden dimension.
        head_dim: Per-head dimension.
        num_attention_heads: Number of attention heads (Q).
        num_kv_heads: Number of KV heads (grouped-query attention).
    """

    layer_index: int
    layer_name: str
    layer_type: str
    total_layers: int = _QWEN3_NEXT_TOTAL_LAYERS
    checkpoint_path: Path = field(default_factory=lambda: Path("."))
    config: object = None  # PretrainedConfig — typed as object to avoid import issues
    device: str = "cpu"
    dtype: torch.dtype = torch.float16
    num_experts: int = _QWEN3_NEXT_NUM_EXPERTS
    num_active_experts: int = _QWEN3_NEXT_NUM_ACTIVE_EXPERTS
    hidden_size: int = _QWEN3_NEXT_HIDDEN_SIZE
    head_dim: int = _QWEN3_NEXT_HEAD_DIM
    num_attention_heads: int = _QWEN3_NEXT_NUM_ATTENTION_HEADS
    num_kv_heads: int = _QWEN3_NEXT_NUM_KV_HEADS

    @property
    def is_deltanet(self) -> bool:
        """True if this layer uses DeltaNet linear attention."""
        return self.layer_type == "linear_attention"

    @property
    def is_rope_enabled(self) -> bool:
        """True if this layer uses RoPE (full attention with rotary embeddings)."""
        return self.layer_type == "full_attention"

    @property
    def layer_fraction(self) -> float:
        """Position of this layer as a fraction of total layers (0.0 to 1.0)."""
        if self.total_layers <= 1:
            return 0.0
        return self.layer_index / (self.total_layers - 1)

    @property
    def safetensors_filename(self) -> str:
        """The safetensors filename for this layer (e.g. 'model.layers.0.safetensors').

        AirLLM naming convention: layer_name already includes trailing dot,
        so filename is layer_name + 'safetensors'.
        """
        return self.layer_name + "safetensors"

    @property
    def safetensors_path(self) -> Path:
        """Full path to this layer's safetensors file."""
        return self.checkpoint_path / self.safetensors_filename

    @property
    def done_marker_path(self) -> Path:
        """Full path to the .done marker file for this layer."""
        return self.checkpoint_path / (self.safetensors_filename + ".done")

    @classmethod
    def from_config(
        cls,
        index: int,
        config,
        checkpoint_path: Path | str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> LayerContext:
        """Build a LayerContext from a HuggingFace PretrainedConfig.

        Extracts architecture parameters from config attributes, falling back
        to Qwen3-Next defaults when attributes are missing.

        Args:
            index: Zero-based layer index.
            config: HuggingFace PretrainedConfig instance.
            checkpoint_path: Path to splitted_model directory containing layer safetensors.
            device: Target device string (default "cpu").
            dtype: Target dtype (default torch.float16).

        Returns:
            Fully populated LayerContext for the given layer.
        """
        checkpoint_path = Path(checkpoint_path)

        # Determine layer type from config or pattern
        layer_type = _get_layer_type(index, config)

        # Extract architecture params from config with Qwen3-Next defaults
        total_layers = getattr(config, "num_hidden_layers", _QWEN3_NEXT_TOTAL_LAYERS)
        num_experts = getattr(config, "num_experts", _QWEN3_NEXT_NUM_EXPERTS)
        num_active_experts = getattr(
            config, "num_activated_experts",
            getattr(config, "num_experts_per_tok", _QWEN3_NEXT_NUM_ACTIVE_EXPERTS),
        )
        hidden_size = getattr(config, "hidden_size", _QWEN3_NEXT_HIDDEN_SIZE)
        head_dim = getattr(config, "head_dim", _QWEN3_NEXT_HEAD_DIM)
        num_attention_heads = getattr(
            config, "num_attention_heads", _QWEN3_NEXT_NUM_ATTENTION_HEADS
        )
        num_kv_heads = getattr(
            config, "num_key_value_heads", _QWEN3_NEXT_NUM_KV_HEADS
        )

        layer_name = f"model.layers.{index}."

        return cls(
            layer_index=index,
            layer_name=layer_name,
            layer_type=layer_type,
            total_layers=total_layers,
            checkpoint_path=checkpoint_path,
            config=config,
            device=device,
            dtype=dtype,
            num_experts=num_experts,
            num_active_experts=num_active_experts,
            hidden_size=hidden_size,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
        )

    def __repr__(self) -> str:
        return (
            f"LayerContext(index={self.layer_index}, "
            f"name={self.layer_name!r}, "
            f"type={self.layer_type!r}, "
            f"deltanet={self.is_deltanet}, "
            f"device={self.device!r})"
        )
