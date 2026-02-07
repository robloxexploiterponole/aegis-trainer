"""
LayerIO — Layer-level file I/O for AEGIS AI Trainer.

Wraps AirLLM's layer loading/saving via composition (NOT inheritance).
Uses safetensors directly for reliable load/save without depending on
AirLLM's persister at runtime.

File naming convention (from AirLLM's SafetensorModelPersister):
  - Layer file: {checkpoint_path}/{layer_name}safetensors
    e.g. splitted_model/model.layers.0.safetensors
  - Done marker: {checkpoint_path}/{layer_name}safetensors.done
  - layer_name already has trailing dot, so "model.layers.0." + "safetensors"

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

import torch
from safetensors.torch import load_file, save_file

from ..layer_context import LayerContext

logger = logging.getLogger(__name__)


class LayerIO:
    """Layer-level file I/O using safetensors format.

    Handles loading, saving, and backing up individual layer state dicts.
    Uses composition rather than inheriting from AirLLM — reads/writes
    safetensors files directly using the same naming convention.

    Usage::

        layer_io = LayerIO(checkpoint_path=Path("/models/qwen3-next/splitted_model"))
        ctx = LayerContext.from_config(0, config, checkpoint_path)

        state_dict = layer_io.load(ctx)
        # ... modify state_dict ...
        layer_io.save(state_dict, ctx, output_path=Path("/output/splitted_model"))
    """

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        backup_dir: Optional[Path] = None,
    ):
        """Initialize LayerIO.

        Args:
            checkpoint_path: Default path to splitted_model directory for loading.
                Can be overridden per-call via LayerContext.checkpoint_path.
            backup_dir: Directory for layer backups. Defaults to
                checkpoint_path / ".aegis_backups" if not specified.
        """
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.backup_dir = Path(backup_dir) if backup_dir else None

    def _resolve_checkpoint_path(self, ctx: LayerContext) -> Path:
        """Get the checkpoint path, preferring ctx over the instance default."""
        path = ctx.checkpoint_path if ctx.checkpoint_path else self.checkpoint_path
        if path is None:
            raise ValueError(
                "No checkpoint_path provided in LayerContext or LayerIO constructor."
            )
        return Path(path)

    def load(
        self,
        ctx: LayerContext,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load a single layer's state dict from disk.

        Reads the safetensors file at {checkpoint_path}/{layer_name}safetensors.

        Args:
            ctx: LayerContext identifying the layer to load.
            device: Device to load tensors onto (default "cpu").

        Returns:
            Dictionary mapping parameter names to tensors.

        Raises:
            FileNotFoundError: If the layer's safetensors file does not exist.
        """
        checkpoint_path = self._resolve_checkpoint_path(ctx)
        layer_file = checkpoint_path / ctx.safetensors_filename

        if not layer_file.exists():
            raise FileNotFoundError(
                f"Layer file not found: {layer_file}. "
                f"Expected at {checkpoint_path / ctx.safetensors_filename}. "
                f"Has the model been split with AirLLM?"
            )

        logger.debug("Loading layer %s from %s", ctx.layer_name, layer_file)
        state_dict = load_file(str(layer_file), device=device)
        logger.info(
            "Loaded layer %s: %d tensors, device=%s",
            ctx.layer_name,
            len(state_dict),
            device,
        )
        return state_dict

    def save(
        self,
        state_dict: dict[str, torch.Tensor],
        ctx: LayerContext,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Save a layer's state dict to disk as safetensors.

        Creates the output directory if needed. Also creates a .done marker
        file to match AirLLM's convention for detecting complete saves.

        Args:
            state_dict: Dictionary mapping parameter names to tensors.
            ctx: LayerContext identifying the layer.
            output_path: Directory to save into. Defaults to ctx.checkpoint_path.

        Returns:
            Path to the saved safetensors file.
        """
        if output_path is None:
            output_path = self._resolve_checkpoint_path(ctx)
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        layer_file = output_path / ctx.safetensors_filename
        done_marker = output_path / (ctx.safetensors_filename + ".done")

        # Remove stale done marker before writing (in case of interrupted save)
        if done_marker.exists():
            done_marker.unlink()

        logger.debug("Saving layer %s to %s", ctx.layer_name, layer_file)

        # Ensure all tensors are contiguous before saving (safetensors requirement)
        contiguous_dict = {}
        for key, tensor in state_dict.items():
            if not tensor.is_contiguous():
                contiguous_dict[key] = tensor.contiguous()
            else:
                contiguous_dict[key] = tensor

        save_file(contiguous_dict, str(layer_file))

        # Write done marker (AirLLM convention — presence means save is complete)
        done_marker.touch()

        logger.info(
            "Saved layer %s: %d tensors to %s",
            ctx.layer_name,
            len(state_dict),
            layer_file,
        )
        return layer_file

    def _backup_layer(
        self,
        ctx: LayerContext,
        backup_dir: Optional[Path] = None,
    ) -> Optional[Path]:
        """Copy the original layer file to a backup directory before modification.

        Args:
            ctx: LayerContext identifying the layer to back up.
            backup_dir: Override backup directory. Defaults to instance backup_dir,
                then falls back to {checkpoint_path}/.aegis_backups/.

        Returns:
            Path to the backup file, or None if the source file doesn't exist.
        """
        checkpoint_path = self._resolve_checkpoint_path(ctx)
        source = checkpoint_path / ctx.safetensors_filename

        if not source.exists():
            logger.warning(
                "Cannot backup layer %s — source file %s does not exist",
                ctx.layer_name,
                source,
            )
            return None

        # Resolve backup directory
        if backup_dir is None:
            backup_dir = self.backup_dir
        if backup_dir is None:
            backup_dir = checkpoint_path / ".aegis_backups"

        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

        dest = backup_dir / ctx.safetensors_filename
        if dest.exists():
            logger.debug(
                "Backup already exists for %s at %s — skipping",
                ctx.layer_name,
                dest,
            )
            return dest

        logger.info("Backing up %s -> %s", source, dest)
        shutil.copy2(str(source), str(dest))

        # Also copy done marker if it exists
        source_marker = checkpoint_path / (ctx.safetensors_filename + ".done")
        if source_marker.exists():
            dest_marker = backup_dir / (ctx.safetensors_filename + ".done")
            shutil.copy2(str(source_marker), str(dest_marker))

        return dest

    def layer_exists(self, ctx: LayerContext) -> bool:
        """Check if a layer's safetensors file exists and has a done marker.

        Args:
            ctx: LayerContext identifying the layer.

        Returns:
            True if both the safetensors file and .done marker exist.
        """
        checkpoint_path = self._resolve_checkpoint_path(ctx)
        safetensors_file = checkpoint_path / ctx.safetensors_filename
        done_marker = checkpoint_path / (ctx.safetensors_filename + ".done")
        return safetensors_file.exists() and done_marker.exists()
