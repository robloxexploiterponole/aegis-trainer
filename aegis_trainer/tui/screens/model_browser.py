"""
ModelBrowserScreen — Browse and inspect available models.

Features:
  - Lists models found in /AEGIS_AI/models/ (and subdirectories)
  - Shows model architecture info: layer count, param count, MoE config
  - Allows navigating into individual layers
  - Color-codes DeltaNet layers (yellow) vs Attention layers (cyan)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import DataTable, RichLog, Static

from aegis_trainer.tui.theme import COLORS

logger = logging.getLogger(__name__)

# Default search paths for models
MODEL_SEARCH_PATHS = [
    Path("/AEGIS_AI/models"),
]


def _find_models(search_paths: list[Path] | None = None) -> list[dict]:
    """Scan directories for HuggingFace-style model folders and GGUF files.

    A directory is considered a model if it contains a config.json file.
    Standalone .gguf files are also detected and listed with metadata
    parsed from filename conventions.

    Returns:
        List of dicts with model metadata: name, path, params, layers, etc.
    """
    paths = search_paths or MODEL_SEARCH_PATHS
    models: list[dict] = []

    for base in paths:
        if not base.exists():
            continue
        # Check if base itself is a model directory
        candidates = [base] if (base / "config.json").exists() else []
        # Also check immediate subdirectories
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir() and (child / "config.json").exists():
                    candidates.append(child)
                elif child.is_dir():
                    # Scan for GGUF files inside subdirectories
                    for gguf in sorted(child.glob("*.gguf")):
                        info = _read_gguf_info(gguf)
                        if info:
                            models.append(info)
            # Also scan base directory itself for GGUF files
            for gguf in sorted(base.glob("*.gguf")):
                info = _read_gguf_info(gguf)
                if info:
                    models.append(info)

        for model_dir in candidates:
            info = _read_model_info(model_dir)
            if info:
                models.append(info)

    return models


def _read_gguf_info(gguf_path: Path) -> dict | None:
    """Extract model metadata from a GGUF filename.

    Parses naming conventions like:
        Qwen3-Coder-Next-UD-Q4_K_XL.gguf
        Qwen3-Next-80B-A3B-Thinking-UD-Q4_K_XL.gguf
    """
    if not gguf_path.exists() or not gguf_path.suffix == ".gguf":
        return None

    name = gguf_path.stem
    size_gb = gguf_path.stat().st_size / (1024 ** 3)

    # Try to parse quantization from filename
    quant = "unknown"
    for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_XL", "Q4_K_M",
              "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS",
              "IQ3_M", "IQ2_M", "F16", "BF16", "F32"):
        if q in name:
            quant = q
            break

    # Try to detect param count from filename
    param_str = "?"
    for token in name.replace("-", " ").split():
        if token.upper().endswith("B") and token[:-1].replace(".", "").isdigit():
            param_str = f"~{token}"
            break

    # Detect MoE pattern (e.g. "A3B" means 3B active)
    num_experts = 0
    num_active = 0
    if "A3B" in name:
        num_experts = 512
        num_active = 10
        if param_str == "?":
            param_str = "~80B"

    # Detect model type hints
    model_type = "gguf"
    if "Qwen3" in name:
        model_type = "qwen3"
    if "Next" in name:
        model_type = "qwen3-next"

    # Guess layer count from known architectures
    num_layers = 0
    if "80B" in name or "A3B" in name:
        num_layers = 48

    return {
        "name": name,
        "path": str(gguf_path),
        "model_type": model_type,
        "architectures": [f"GGUF ({quant})"],
        "num_layers": num_layers,
        "hidden_size": 2048 if num_layers == 48 else 0,
        "num_experts": num_experts,
        "num_active_experts": num_active,
        "num_attention_heads": 16 if num_layers == 48 else 0,
        "num_kv_heads": 2 if num_layers == 48 else 0,
        "vocab_size": 0,
        "intermediate_size": 0,
        "param_estimate": param_str,
        "layer_types": [],
        "format": "gguf",
        "quant": quant,
        "size_gb": round(size_gb, 1),
    }


def _read_model_info(model_dir: Path) -> dict | None:
    """Read config.json and extract model metadata."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read %s: %s", config_path, exc)
        return None

    # Extract key architecture parameters
    num_layers = config.get("num_hidden_layers", 0)
    hidden_size = config.get("hidden_size", 0)
    num_experts = config.get("num_experts", 0)
    num_active_experts = config.get(
        "num_activated_experts",
        config.get("num_experts_per_tok", 0),
    )
    num_attention_heads = config.get("num_attention_heads", 0)
    num_kv_heads = config.get("num_key_value_heads", 0)
    model_type = config.get("model_type", "unknown")
    architectures = config.get("architectures", [])
    vocab_size = config.get("vocab_size", 0)
    intermediate_size = config.get("intermediate_size", 0)

    # Rough parameter count estimate (for display)
    # This is approximate: embed + layers * (attn + ff + experts)
    param_estimate = _estimate_params(config)

    # Layer types if available
    layer_types = config.get("layer_types", [])

    return {
        "name": model_dir.name,
        "path": str(model_dir),
        "model_type": model_type,
        "architectures": architectures,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_experts": num_experts,
        "num_active_experts": num_active_experts,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "param_estimate": param_estimate,
        "layer_types": layer_types,
    }


def _estimate_params(config: dict) -> str:
    """Rough parameter count estimate for display purposes."""
    hidden = config.get("hidden_size", 0)
    layers = config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)
    intermediate = config.get("intermediate_size", 0)
    num_experts = config.get("num_experts", 1)

    if hidden == 0 or layers == 0:
        return "?"

    # Embedding
    embed_params = vocab * hidden * 2  # input + output embeddings

    # Per-layer rough estimate
    attn_params = 4 * hidden * hidden  # Q, K, V, O projections
    ff_params = 2 * hidden * intermediate * max(num_experts, 1)
    layer_params = attn_params + ff_params

    total = embed_params + layers * layer_params

    if total > 1e12:
        return f"~{total / 1e12:.0f}T"
    if total > 1e9:
        return f"~{total / 1e9:.0f}B"
    if total > 1e6:
        return f"~{total / 1e6:.0f}M"
    return f"~{total:.0f}"


class ModelBrowserScreen(Widget):
    """Browse available models and view architecture details."""

    DEFAULT_CSS = """
    ModelBrowserScreen {
        height: 1fr;
        layout: horizontal;
    }
    ModelBrowserScreen .mb-list {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    ModelBrowserScreen .mb-detail {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        border-left: solid #2a3a4a;
        padding: 0 1;
    }
    ModelBrowserScreen .mb-title {
        color: #00d4ff;
        text-style: bold;
        padding: 1 0;
    }
    ModelBrowserScreen DataTable {
        height: 1fr;
        background: #131924;
    }
    ModelBrowserScreen RichLog {
        height: 1fr;
        background: #131924;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._models: list[dict] = []

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(classes="mb-list"):
                yield Static("Available Models", classes="mb-title")
                yield DataTable(id="mb-model-table", cursor_type="row")

            with Vertical(classes="mb-detail"):
                yield Static("Model Details", classes="mb-title")
                yield RichLog(highlight=True, markup=True, id="mb-detail-log")

    def on_mount(self) -> None:
        """Scan for models and populate the table."""
        table = self.query_one("#mb-model-table", DataTable)
        table.add_columns("Name", "Type", "Layers", "Params", "Experts")

        self._models = _find_models()

        if not self._models:
            detail = self.query_one("#mb-detail-log", RichLog)
            detail.write(
                f"[{COLORS['text_muted']}]No models found in search paths.[/]"
            )
            detail.write(
                f"[{COLORS['text_muted']}]Expected: /AEGIS_AI/models/<model_name>/config.json[/]"
            )
            return

        for model in self._models:
            experts_str = ""
            if model["num_experts"] > 0:
                experts_str = f"{model['num_active_experts']}/{model['num_experts']}"
            table.add_row(
                model["name"],
                model["model_type"],
                str(model["num_layers"]),
                model["param_estimate"],
                experts_str,
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Display details for the selected model."""
        row_index = event.cursor_row
        if row_index < 0 or row_index >= len(self._models):
            return

        model = self._models[row_index]
        detail = self.query_one("#mb-detail-log", RichLog)
        detail.clear()

        detail.write(f"[bold {COLORS['accent_cyan']}]{model['name']}[/]")
        detail.write(f"  Path: {model['path']}")
        detail.write(f"  Type: {model['model_type']}")
        if model["architectures"]:
            detail.write(f"  Architectures: {', '.join(model['architectures'])}")
        detail.write(f"  Parameters: {model['param_estimate']}")

        # GGUF-specific info
        if model.get("format") == "gguf":
            detail.write(f"  Quantization: {model.get('quant', '?')}")
            detail.write(f"  File size: {model.get('size_gb', 0):.1f} GB")
        detail.write("")
        detail.write(f"[bold]Architecture:[/]")
        detail.write(f"  Hidden size:    {model['hidden_size']}")
        detail.write(f"  Layers:         {model['num_layers']}")
        detail.write(f"  Attention heads: {model['num_attention_heads']}")
        detail.write(f"  KV heads:       {model['num_kv_heads']}")
        detail.write(f"  Vocab size:     {model['vocab_size']}")
        detail.write(f"  Intermediate:   {model['intermediate_size']}")

        if model["num_experts"] > 0:
            detail.write("")
            detail.write(f"[bold]Mixture of Experts:[/]")
            detail.write(f"  Total experts:  {model['num_experts']}")
            detail.write(f"  Active per tok: {model['num_active_experts']}")

        # Layer type breakdown
        layer_types = model.get("layer_types", [])
        if layer_types:
            detail.write("")
            detail.write(f"[bold]Layer Types:[/]")
            deltanet_count = sum(1 for t in layer_types if t == "linear_attention")
            attn_count = sum(1 for t in layer_types if t == "full_attention")
            detail.write(
                f"  [{COLORS['accent_yellow']}]DeltaNet (linear_attention):[/] {deltanet_count}"
            )
            detail.write(
                f"  [{COLORS['accent_cyan']}]Full Attention:[/] {attn_count}"
            )

            detail.write("")
            detail.write(f"[bold]Layer Map:[/]")
            # Show layer pattern in rows of 16
            row: list[str] = []
            for i, lt in enumerate(layer_types):
                if lt == "linear_attention":
                    row.append(f"[{COLORS['accent_yellow']}]D[/]")
                else:
                    row.append(f"[{COLORS['accent_cyan']}]A[/]")
                if (i + 1) % 16 == 0:
                    detail.write(f"  {'  '.join(row)}")
                    row = []
            if row:
                detail.write(f"  {'  '.join(row)}")
        elif model["num_layers"] == 48:
            # Assume Qwen3-Next default pattern
            detail.write("")
            detail.write(f"[bold]Layer Types (default Qwen3-Next pattern):[/]")
            detail.write(
                f"  [{COLORS['accent_yellow']}]DeltaNet (linear_attention):[/] 36"
            )
            detail.write(
                f"  [{COLORS['accent_cyan']}]Full Attention:[/] 12"
            )
            detail.write(f"  Pattern: 3x DeltaNet + 1x Attention, repeated 12 times")
