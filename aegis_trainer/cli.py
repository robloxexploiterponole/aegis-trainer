"""
CLI entry point for AEGIS AI Trainer.

Click-based command-line interface providing:
  - aegis-trainer run abliterate   — Directional ablation (deregulation)
  - aegis-trainer run longrope     — Context window extension
  - aegis-trainer run lora         — LoRA adapter merge/train
  - aegis-trainer run quantize     — Model quantization
  - aegis-trainer inspect          — Model architecture and tensor inspection
  - aegis-trainer queue            — Queue management (list/cancel/clear)
  - aegis-trainer status           — System resource summary
  - aegis-trainer tui              — Launch the Textual TUI

Console script entry point (setup.py):
  aegis-trainer = aegis_trainer.cli:main

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from aegis_trainer import __version__
from aegis_trainer.tui.theme import COLORS, header_rich

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main group
# ---------------------------------------------------------------------------

@click.group(invoke_without_command=True)
@click.version_option(version=__version__, prog_name="aegis-trainer")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """AEGIS AI Trainer -- Layer-by-layer model training and modification.

    Uses AirLLM layer streaming to modify 80B+ parameter MoE models
    on consumer GPUs without loading the full model into memory.
    """
    # Configure logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # If no subcommand, print help
    if ctx.invoked_subcommand is None:
        console.print(header_rich())
        console.print()
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Run subgroup
# ---------------------------------------------------------------------------

@main.group()
@click.pass_context
def run(ctx: click.Context) -> None:
    """Run a training/modification operation."""
    pass


# ---------------------------------------------------------------------------
# run abliterate
# ---------------------------------------------------------------------------

@run.command()
@click.option("--model", required=True, type=click.Path(exists=False), help="Path to source model or HuggingFace model ID.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for modified model.")
@click.option("--directions", type=click.Path(exists=True), default=None, help="Path to pre-computed refusal directions file.")
@click.option("--device-map", default="auto", show_default=True, help="Device mapping strategy.")
@click.option("--layers", default="all", show_default=True, help="Layers to process (e.g. 'all', '0-23', '12,15,18').")
@click.pass_context
def abliterate(
    ctx: click.Context,
    model: str,
    output: str,
    directions: str | None,
    device_map: str,
    layers: str,
) -> None:
    """Run abliteration (deregulation via directional ablation).

    Removes refusal behavior by computing and ablating directional components
    from model weights, layer by layer.
    """
    console.print(header_rich())
    console.print()

    config = {
        "operation": "abliterate",
        "model": model,
        "output": output,
        "directions": directions,
        "device_map": device_map,
        "layers": layers,
    }

    _print_operation_config(config)
    _confirm_and_run(config)


# ---------------------------------------------------------------------------
# run longrope
# ---------------------------------------------------------------------------

@run.command()
@click.option("--model", required=True, type=click.Path(exists=False), help="Path to source model or HuggingFace model ID.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for modified model.")
@click.option("--target-context", required=True, type=int, default=524288, show_default=True, help="Target context length in tokens.")
@click.option("--search-method", default="evolutionary", show_default=True, help="RoPE parameter search method.")
@click.pass_context
def longrope(
    ctx: click.Context,
    model: str,
    output: str,
    target_context: int,
    search_method: str,
) -> None:
    """Run LongRoPE context window extension.

    Extends the model's effective context window by optimizing rotary
    position embedding parameters for each attention layer.
    """
    console.print(header_rich())
    console.print()

    config = {
        "operation": "longrope",
        "model": model,
        "output": output,
        "target_context": target_context,
        "search_method": search_method,
    }

    _print_operation_config(config)
    _confirm_and_run(config)


# ---------------------------------------------------------------------------
# run lora
# ---------------------------------------------------------------------------

@run.command()
@click.option("--model", required=True, type=click.Path(exists=False), help="Path to base model or HuggingFace model ID.")
@click.option("--adapter", required=True, type=click.Path(exists=False), help="Path to LoRA adapter directory.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for merged/trained model.")
@click.option("--merge", is_flag=True, default=False, help="Merge adapter into base weights.")
@click.option("--train", is_flag=True, default=False, help="Train a new LoRA adapter.")
@click.option("--rank", type=int, default=32, show_default=True, help="LoRA rank (dimension of low-rank matrices).")
@click.pass_context
def lora(
    ctx: click.Context,
    model: str,
    adapter: str,
    output: str,
    merge: bool,
    train: bool,
    rank: int,
) -> None:
    """Run LoRA adapter merge or training.

    Merge an existing LoRA adapter into base model weights, or train
    a new adapter using layer-by-layer streaming.
    """
    console.print(header_rich())
    console.print()

    if merge and train:
        console.print(f"[{COLORS['accent_red']}]Error: --merge and --train are mutually exclusive.[/]")
        sys.exit(1)

    config = {
        "operation": "lora",
        "model": model,
        "adapter": adapter,
        "output": output,
        "merge": merge,
        "train": train,
        "rank": rank,
    }

    _print_operation_config(config)
    _confirm_and_run(config)


# ---------------------------------------------------------------------------
# run quantize
# ---------------------------------------------------------------------------

@run.command()
@click.option("--model", required=True, type=click.Path(exists=False), help="Path to source model or HuggingFace model ID.")
@click.option("--output", required=True, type=click.Path(), help="Output directory for quantized model.")
@click.option("--quant-type", required=True, default="Q4_K_M", show_default=True, help="Quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0).")
@click.pass_context
def quantize(
    ctx: click.Context,
    model: str,
    output: str,
    quant_type: str,
) -> None:
    """Run model quantization.

    Quantize model weights to a lower precision format for reduced
    memory usage and faster inference.
    """
    console.print(header_rich())
    console.print()

    config = {
        "operation": "quantize",
        "model": model,
        "output": output,
        "quant_type": quant_type,
    }

    _print_operation_config(config)
    _confirm_and_run(config)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", required=True, type=click.Path(exists=False), help="Path to model directory.")
@click.option("--layer", "-l", type=int, default=None, help="Specific layer index to inspect.")
@click.option("--tensor", "-t", type=str, default=None, help="Specific tensor name to inspect.")
def inspect(model: str, layer: int | None, tensor: str | None) -> None:
    """Inspect model architecture, layers, and tensors.

    Without --layer: prints architecture summary and layer type map.
    With --layer N: prints detailed info for layer N.
    With --tensor NAME: prints statistics for a specific tensor.
    """
    console.print(header_rich())
    console.print()

    model_path = Path(model)
    config_path = model_path / "config.json"

    if not config_path.exists():
        console.print(
            f"[{COLORS['accent_red']}]Error: config.json not found at {config_path}[/]"
        )
        console.print(
            f"[{COLORS['text_muted']}]Provide the model directory containing config.json.[/]"
        )
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[{COLORS['accent_red']}]Error reading config.json: {exc}[/]")
        sys.exit(1)

    if layer is not None:
        _inspect_layer(config, model_path, layer, tensor)
    elif tensor is not None:
        console.print(
            f"[{COLORS['accent_yellow']}]--tensor requires --layer to be specified.[/]"
        )
        sys.exit(1)
    else:
        _inspect_model_summary(config, model_path)


def _inspect_model_summary(config: dict, model_path: Path) -> None:
    """Print model architecture summary."""
    from aegis_trainer.layer_context import _get_layer_type

    num_layers = config.get("num_hidden_layers", 0)
    hidden_size = config.get("hidden_size", 0)
    num_experts = config.get("num_experts", 0)
    num_active = config.get("num_activated_experts", config.get("num_experts_per_tok", 0))
    num_heads = config.get("num_attention_heads", 0)
    num_kv = config.get("num_key_value_heads", 0)
    model_type = config.get("model_type", "unknown")
    vocab_size = config.get("vocab_size", 0)

    # Architecture table
    table = Table(title="Model Architecture", border_style=COLORS["border"])
    table.add_column("Property", style=COLORS["accent_cyan"])
    table.add_column("Value", style=COLORS["text"])

    table.add_row("Path", str(model_path))
    table.add_row("Type", model_type)
    table.add_row("Layers", str(num_layers))
    table.add_row("Hidden size", str(hidden_size))
    table.add_row("Attention heads", str(num_heads))
    table.add_row("KV heads", str(num_kv))
    table.add_row("Vocab size", f"{vocab_size:,}")

    if num_experts > 0:
        table.add_row("Total experts", str(num_experts))
        table.add_row("Active experts", str(num_active))

    console.print(table)
    console.print()

    # Layer type map
    if num_layers > 0:
        layer_types = config.get("layer_types", [])
        deltanet_count = 0
        attention_count = 0

        console.print(f"[bold {COLORS['accent_cyan']}]Layer Map:[/]")

        row_cells: list[str] = []
        for i in range(num_layers):
            if layer_types and i < len(layer_types):
                lt = layer_types[i]
            else:
                lt = _get_layer_type(i)

            if lt == "linear_attention":
                deltanet_count += 1
                row_cells.append(f"[{COLORS['accent_yellow']}]D[/]")
            else:
                attention_count += 1
                row_cells.append(f"[{COLORS['accent_cyan']}]A[/]")

            if (i + 1) % 16 == 0:
                row_indices = "  ".join(f"[dim]{j:2d}[/]" for j in range(i - 15, i + 1))
                row_blocks = "  ".join(row_cells)
                console.print(f"  {row_indices}")
                console.print(f"  {row_blocks}")
                console.print()
                row_cells = []

        if row_cells:
            start = num_layers - len(row_cells)
            row_indices = "  ".join(f"[dim]{j:2d}[/]" for j in range(start, num_layers))
            row_blocks = "  ".join(row_cells)
            console.print(f"  {row_indices}")
            console.print(f"  {row_blocks}")
            console.print()

        console.print(
            f"  [{COLORS['accent_yellow']}]D[/]=DeltaNet ({deltanet_count})  "
            f"[{COLORS['accent_cyan']}]A[/]=Attention ({attention_count})  "
            f"Total: {num_layers}"
        )


def _inspect_layer(
    config: dict,
    model_path: Path,
    layer_idx: int,
    tensor_name: str | None,
) -> None:
    """Print detailed info for a specific layer."""
    from aegis_trainer.layer_context import _get_layer_type

    num_layers = config.get("num_hidden_layers", 48)
    if layer_idx < 0 or layer_idx >= num_layers:
        console.print(
            f"[{COLORS['accent_red']}]Error: Layer {layer_idx} out of range (0-{num_layers - 1}).[/]"
        )
        sys.exit(1)

    layer_types = config.get("layer_types", [])
    if layer_types and layer_idx < len(layer_types):
        lt = layer_types[layer_idx]
    else:
        lt = _get_layer_type(layer_idx)

    is_deltanet = lt == "linear_attention"
    kind_str = "DeltaNet (linear_attention)" if is_deltanet else "Full Attention"
    color = COLORS["accent_yellow"] if is_deltanet else COLORS["accent_cyan"]

    console.print(f"[bold {color}]Layer {layer_idx}: {kind_str}[/]")
    console.print(f"  Prefix: model.layers.{layer_idx}.")
    console.print()

    # Try to load actual safetensors
    safetensors_file = model_path / f"model.layers.{layer_idx}.safetensors"
    if safetensors_file.exists():
        try:
            from safetensors import safe_open

            table = Table(title=f"Layer {layer_idx} Tensors", border_style=COLORS["border"])
            table.add_column("Tensor", style=COLORS["text"])
            table.add_column("Shape", style=COLORS["accent_cyan"])
            table.add_column("Dtype", style=COLORS["text_muted"])
            table.add_column("Size (MB)", justify="right", style=COLORS["text"])

            with safe_open(str(safetensors_file), framework="pt") as f:
                for key in sorted(f.keys()):
                    t = f.get_tensor(key)
                    shape_str = str(list(t.shape))
                    size_mb = t.nelement() * t.element_size() / 1e6

                    if tensor_name and tensor_name in key:
                        # Show detailed stats for this tensor
                        table.add_row(
                            f"[bold]{key}[/]", shape_str, str(t.dtype), f"{size_mb:.2f}"
                        )
                        console.print(table)
                        console.print()
                        _print_tensor_stats(key, t)
                        return
                    else:
                        table.add_row(key, shape_str, str(t.dtype), f"{size_mb:.2f}")

            console.print(table)

        except ImportError:
            console.print(
                f"[{COLORS['accent_yellow']}]safetensors package required for tensor inspection.[/]"
            )
        except Exception as exc:
            console.print(f"[{COLORS['accent_red']}]Error loading safetensors: {exc}[/]")
    else:
        console.print(
            f"[{COLORS['text_muted']}]Safetensors file not found: {safetensors_file}[/]"
        )
        console.print(
            f"[{COLORS['text_muted']}]Run with a model that has been split into per-layer files.[/]"
        )


def _print_tensor_stats(name: str, tensor) -> None:
    """Print detailed statistics for a tensor."""
    import torch

    table = Table(title=f"Tensor Statistics: {name}", border_style=COLORS["border"])
    table.add_column("Statistic", style=COLORS["accent_cyan"])
    table.add_column("Value", style=COLORS["text"])

    t_float = tensor.float()

    table.add_row("Shape", str(list(tensor.shape)))
    table.add_row("Dtype", str(tensor.dtype))
    table.add_row("Elements", f"{tensor.nelement():,}")
    table.add_row("Size (MB)", f"{tensor.nelement() * tensor.element_size() / 1e6:.2f}")
    table.add_row("Min", f"{t_float.min().item():.6f}")
    table.add_row("Max", f"{t_float.max().item():.6f}")
    table.add_row("Mean", f"{t_float.mean().item():.6f}")
    table.add_row("Std", f"{t_float.std().item():.6f}")

    # Sparsity
    zero_count = (tensor == 0).sum().item()
    sparsity = zero_count / tensor.nelement() * 100
    table.add_row("Zeros", f"{zero_count:,} ({sparsity:.1f}%)")

    # NaN/Inf check
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    if nan_count > 0:
        table.add_row("NaN values", f"[bold red]{nan_count:,}[/]")
    if inf_count > 0:
        table.add_row("Inf values", f"[bold red]{inf_count:,}[/]")
    if nan_count == 0 and inf_count == 0:
        table.add_row("Health", f"[{COLORS['accent_green']}]Clean (no NaN/Inf)[/]")

    console.print(table)


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------

@main.command()
@click.option("--list", "list_queue", is_flag=True, default=False, help="List all queued jobs.")
@click.option("--cancel", type=int, default=None, help="Cancel job by index.")
@click.option("--clear", is_flag=True, default=False, help="Clear all completed/cancelled jobs.")
def queue(list_queue: bool, cancel: int | None, clear: bool) -> None:
    """Manage the operation queue."""
    console.print(header_rich())
    console.print()

    if not list_queue and cancel is None and not clear:
        # Default to listing
        list_queue = True

    if list_queue:
        console.print(f"[bold {COLORS['accent_cyan']}]Operation Queue[/]")
        console.print()

        table = Table(border_style=COLORS["border"])
        table.add_column("#", style=COLORS["text_muted"], justify="right")
        table.add_column("Operation", style=COLORS["accent_cyan"])
        table.add_column("Model", style=COLORS["text"])
        table.add_column("Status", style=COLORS["text"])
        table.add_column("Progress", justify="right", style=COLORS["text"])

        # In production, this would read from the QQMS queue
        console.print(table)
        console.print(
            f"[{COLORS['text_muted']}]Queue is empty. Use 'aegis-trainer run' or the TUI to queue operations.[/]"
        )

    if cancel is not None:
        console.print(
            f"[{COLORS['accent_yellow']}]Cancelling job #{cancel}...[/]"
        )
        # In production, this would call into the QQMS queue
        console.print(
            f"[{COLORS['text_muted']}]Queue integration pending. Use the TUI for queue management.[/]"
        )

    if clear:
        console.print(
            f"[{COLORS['accent_yellow']}]Clearing completed and cancelled jobs...[/]"
        )
        console.print(
            f"[{COLORS['text_muted']}]Queue integration pending. Use the TUI for queue management.[/]"
        )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@main.command()
def status() -> None:
    """Print system resources and active operation status."""
    from aegis_trainer.utils.resource_monitor import ResourceMonitor

    console.print(header_rich())
    console.print()

    monitor = ResourceMonitor()
    snap = monitor.get_snapshot()

    # System resources table
    table = Table(title="System Resources", border_style=COLORS["border"])
    table.add_column("Resource", style=COLORS["accent_cyan"])
    table.add_column("Usage", style=COLORS["text"])
    table.add_column("Bar", style=COLORS["text"])

    # CPU
    cpu_bar = _make_bar(snap.cpu_percent)
    table.add_row("CPU", f"{snap.cpu_percent:.1f}%", cpu_bar)

    # RAM
    ram_gb_used = snap.ram_used_bytes / (1024 ** 3)
    ram_gb_total = snap.ram_total_bytes / (1024 ** 3)
    ram_bar = _make_bar(snap.ram_percent)
    table.add_row(
        "RAM",
        f"{ram_gb_used:.1f} / {ram_gb_total:.1f} GB ({snap.ram_percent:.1f}%)",
        ram_bar,
    )

    # VRAM
    if snap.vram_total_bytes > 0:
        vram_gb_used = snap.vram_used_bytes / (1024 ** 3)
        vram_gb_total = snap.vram_total_bytes / (1024 ** 3)
        vram_bar = _make_bar(snap.vram_percent)
        table.add_row(
            "VRAM",
            f"{vram_gb_used:.1f} / {vram_gb_total:.1f} GB ({snap.vram_percent:.1f}%)",
            vram_bar,
        )
    else:
        table.add_row("VRAM", "N/A (no GPU monitoring)", "")

    console.print(table)
    console.print()

    # Active operation
    console.print(f"[bold {COLORS['accent_cyan']}]Active Operation:[/]")
    console.print(f"  [{COLORS['text_muted']}]No active operation.[/]")
    console.print()

    # Hardware info
    import psutil

    console.print(f"[bold {COLORS['accent_cyan']}]Hardware:[/]")
    console.print(f"  CPU cores: {psutil.cpu_count(logical=True)}")
    console.print(f"  RAM total: {snap.ram_total_bytes / (1024**3):.1f} GB")
    if snap.vram_total_bytes > 0:
        console.print(f"  VRAM total: {snap.vram_total_bytes / (1024**3):.1f} GB")


def _make_bar(percent: float, width: int = 30) -> str:
    """Create a colored progress bar string for Rich."""
    from aegis_trainer.tui.theme import resource_color

    color = resource_color(percent)
    filled = int(width * min(percent, 100.0) / 100.0)
    empty = width - filled

    return f"[{color}]{'█' * filled}[/][dim]{'░' * empty}[/]"


# ---------------------------------------------------------------------------
# tui
# ---------------------------------------------------------------------------

@main.command()
@click.option("--model", type=click.Path(exists=False), default=None, help="Pre-select a model path for the inspector.")
def tui(model: str | None) -> None:
    """Launch the AEGIS AI Trainer Terminal User Interface."""
    from aegis_trainer.tui.app import AegisTrainerApp

    app = AegisTrainerApp(model_path=model)
    app.run()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_operation_config(config: dict) -> None:
    """Print a Rich-formatted operation configuration summary."""
    op = config.get("operation", "unknown")
    table = Table(title=f"Operation: {op}", border_style=COLORS["border"])
    table.add_column("Parameter", style=COLORS["accent_cyan"])
    table.add_column("Value", style=COLORS["text"])

    for key, value in config.items():
        if key == "operation":
            continue
        display_val = str(value) if value is not None else "(not set)"
        table.add_row(key, display_val)

    console.print(table)
    console.print()


def _confirm_and_run(config: dict) -> None:
    """Ask for confirmation and dispatch to the operation runner.

    In production, this queues the operation into the QQMS system.
    Currently prints a placeholder message.
    """
    if not click.confirm("Queue this operation?", default=True):
        console.print(f"[{COLORS['accent_yellow']}]Cancelled.[/]")
        return

    console.print(
        f"[{COLORS['accent_green']}]Operation queued: {config['operation']}[/]"
    )
    console.print(
        f"[{COLORS['text_muted']}]Run 'aegis-trainer status' or 'aegis-trainer tui' to monitor progress.[/]"
    )

    logger.info(
        "Queued operation: %s model=%s output=%s",
        config.get("operation"),
        config.get("model"),
        config.get("output"),
    )


if __name__ == "__main__":
    main()
