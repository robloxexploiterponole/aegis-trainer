"""
LayerTrainer — Core orchestrator for AEGIS AI Trainer.

Iterates all model layers one-at-a-time (following AirLLM's streaming pattern),
applies queued operations to each layer's weights, validates results, and saves
modified weights to the output path. Never loads more than one layer into memory.

Supports:
  - Crash-safe checkpointing (skip already-completed layers on resume)
  - Resource-aware throttling via QQMS queue
  - Dry-run mode for previewing operations without writes
  - Progress callbacks for TUI/CLI integration
  - Per-operation validation with NaN/Inf checking
  - Dead letter queue for failed operations

Architecture:
  Qwen3-Next: 48 layers (36 DeltaNet + 12 full_attention)
  Pattern: 3 linear_attention + 1 full_attention, repeating 12 times
  Weight modification on CPU (layers too large for 11GB VRAM)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import copy
import gc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

from aegis_trainer.layer_context import LayerContext
from aegis_trainer.ops.base import LayerOperation
from aegis_trainer.queue.dlq import DLQEntry
from aegis_trainer.queue.qqms import QQMSConfig, QQMSQueue, _clean_memory
from aegis_trainer.queue.queue_item import QueueItem
from aegis_trainer.utils.checkpoint import CheckpointManager
from aegis_trainer.utils.layer_io import LayerIO
from aegis_trainer.utils.profiler import TrainerProfiler
from aegis_trainer.utils.resource_monitor import ResourceLimits, ResourceMonitor

logger = logging.getLogger(__name__)


@dataclass
class ProgressUpdate:
    """Progress update sent to the progress callback during training.

    Provides enough information for TUI/CLI display: current position,
    resource usage, timing, and ETA.

    Attributes:
        operation_type: Name of the current operation (e.g. "abliteration").
        current_layer: Zero-based index of the layer being processed.
        total_layers: Total number of transformer layers.
        layer_type: "DeltaNet" for linear_attention, "Attention" for full_attention.
        substep: Current substep within layer processing.
        substep_progress: Substep completion fraction (0.0 to 1.0).
        ram_used_gb: Current RAM usage in GB.
        ram_total_gb: Total system RAM in GB.
        vram_used_gb: Current VRAM usage in GB.
        vram_total_gb: Total VRAM in GB.
        cpu_percent: Current CPU utilization percentage.
        elapsed_seconds: Total elapsed wall-clock time.
        eta_seconds: Estimated time remaining, or None if unknown.
        layers_per_minute: Current processing rate.
        timestamp: Unix timestamp of this update.
    """

    operation_type: str
    current_layer: int
    total_layers: int
    layer_type: str  # "DeltaNet" or "Attention"
    substep: str  # "loading" | "modifying" | "saving" | "verifying"
    substep_progress: float  # 0.0-1.0
    ram_used_gb: float
    ram_total_gb: float
    vram_used_gb: float
    vram_total_gb: float
    cpu_percent: float
    elapsed_seconds: float
    eta_seconds: float | None
    layers_per_minute: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainerResult:
    """Summary of a completed training run.

    Attributes:
        completed_layers: Layer names that were successfully processed.
        skipped_layers: Layer names that were skipped (already checkpointed).
        warnings: List of (LayerContext, LayerOperation, message) tuples
            for non-fatal issues encountered during processing.
        dlq_entries: Dead letter queue entries for operations that failed
            after exhausting all retries.
        total_time: Total wall-clock seconds for the run.
    """

    completed_layers: list[str] = field(default_factory=list)
    skipped_layers: list[str] = field(default_factory=list)
    warnings: list[tuple] = field(default_factory=list)
    dlq_entries: list[DLQEntry] = field(default_factory=list)
    total_time: float = 0.0

    def summary(self) -> str:
        """Generate a human-readable summary of the training run.

        Returns:
            Multi-line summary string suitable for logging or display.
        """
        lines = [
            "=" * 60,
            "AEGIS AI Trainer — Run Summary",
            "=" * 60,
            f"  Completed layers: {len(self.completed_layers)}",
            f"  Skipped layers:   {len(self.skipped_layers)}",
            f"  Warnings:         {len(self.warnings)}",
            f"  DLQ failures:     {len(self.dlq_entries)}",
            f"  Total time:       {self.total_time:.1f}s "
            f"({self.total_time / 60:.1f}min)",
        ]

        if self.completed_layers:
            rate = len(self.completed_layers) / max(self.total_time, 0.001) * 60
            lines.append(f"  Processing rate:  {rate:.1f} layers/min")

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")
            for ctx, op, msg in self.warnings[:10]:
                layer = getattr(ctx, 'layer_name', '?') if ctx else '?'
                op_name = type(op).__name__ if op else '?'
                lines.append(f"  [{layer}] {op_name}: {msg}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")

        if self.dlq_entries:
            lines.append("")
            lines.append("Dead Letter Queue:")
            for entry in self.dlq_entries[:10]:
                lines.append(
                    f"  [{entry.layer_name}] {entry.operation_name}: "
                    f"{entry.reason}"
                )
            if len(self.dlq_entries) > 10:
                lines.append(f"  ... and {len(self.dlq_entries) - 10} more")

        lines.append("=" * 60)
        return "\n".join(lines)


class LayerTrainer:
    """Core AEGIS AI Trainer orchestrator.

    Iterates all model layers one at a time, applies operations via the
    QQMS queue, and saves modified weights. Follows AirLLM's forward()
    pattern: load one layer -> process -> free memory -> never hold more
    than one layer.

    Usage::

        trainer = LayerTrainer(
            model_path="/models/Qwen3-Next-80B-A3B",
            operations=[Abliteration(), LongRoPE(target_len=65536)],
            output_path="/output/modified_model",
        )
        result = trainer.run()
        print(result.summary())

    Args:
        model_path: Path to the pretrained model (HuggingFace format).
        operations: List of LayerOperation instances to apply.
        output_path: Directory for writing modified layer weights.
        resource_limits: Resource thresholds for throttling.
        queue_config: QQMS queue configuration.
        enable_validation: Run validation after each operation.
        enable_profiling: Collect detailed timing data.
        dry_run: Preview operations without writing to disk.
        progress_callback: Callable receiving ProgressUpdate objects.
        weight_callback: Callable receiving (state_dict, LayerContext, phase)
            where phase is "before" or "after" modification. Used by the
            TUI Weight Visualizer screen for live visualization.
    """

    def __init__(
        self,
        model_path: str | Path,
        operations: list[LayerOperation],
        output_path: str | Path,
        resource_limits: ResourceLimits | None = None,
        queue_config: QQMSConfig | None = None,
        enable_validation: bool = True,
        enable_profiling: bool = False,
        dry_run: bool = False,
        progress_callback: Callable[[ProgressUpdate], None] | None = None,
        weight_callback: Callable[[dict, "LayerContext", str], None] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.operations = operations
        self.output_path = Path(output_path)
        self.enable_validation = enable_validation
        self.enable_profiling = enable_profiling
        self.dry_run = dry_run
        self.progress_callback = progress_callback
        self.weight_callback = weight_callback

        # Initialize subsystems
        self._resource_monitor = ResourceMonitor(limits=resource_limits)
        self._queue = QQMSQueue(config=queue_config)
        self._queue.bind_resource_monitor(self._resource_monitor)
        self._layer_io = LayerIO()
        self._checkpoint = CheckpointManager(output_path=self.output_path)
        self._profiler = TrainerProfiler() if enable_profiling else None

        # Model config (loaded lazily in run())
        self._config = None
        self._layer_types: list[str] = []
        self._total_layers: int = 0

        logger.info(
            "LayerTrainer initialized: model=%s, output=%s, "
            "ops=%d, validation=%s, profiling=%s, dry_run=%s",
            self.model_path, self.output_path,
            len(self.operations), enable_validation, enable_profiling, dry_run,
        )

    def _load_model_config(self) -> None:
        """Load model configuration from model_path via AutoConfig.

        Extracts layer count, layer types, and architecture parameters
        from the pretrained config.

        Raises:
            FileNotFoundError: If config.json is not found at model_path.
            ImportError: If transformers is not installed.
        """
        from transformers import AutoConfig

        logger.info("Loading model config from %s", self.model_path)
        self._config = AutoConfig.from_pretrained(str(self.model_path))

        # Extract total layers
        self._total_layers = getattr(
            self._config, 'num_hidden_layers', 48
        )

        # Extract layer types (Qwen3-Next pattern)
        config_layer_types = getattr(self._config, 'layer_types', [])
        if config_layer_types:
            self._layer_types = list(config_layer_types)
            logger.info(
                "Layer types from config: %d linear_attention, %d full_attention",
                sum(1 for lt in self._layer_types if lt == 'linear_attention'),
                sum(1 for lt in self._layer_types if lt == 'full_attention'),
            )
        else:
            # Generate default Qwen3-Next pattern: 3 linear + 1 full, x12
            pattern = ["linear_attention"] * 3 + ["full_attention"]
            self._layer_types = (pattern * 12)[:self._total_layers]
            logger.info(
                "Using default Qwen3-Next layer pattern for %d layers",
                self._total_layers,
            )

        logger.info(
            "Model config loaded: %d layers, hidden_size=%d",
            self._total_layers,
            getattr(self._config, 'hidden_size', 'unknown'),
        )

    def _build_layer_contexts(
        self,
        checkpoint_path: Path,
    ) -> list[LayerContext]:
        """Build LayerContext objects for all transformer layers.

        Args:
            checkpoint_path: Path to the splitted model directory.

        Returns:
            List of LayerContext objects, one per transformer layer.
        """
        contexts: list[LayerContext] = []

        for i in range(self._total_layers):
            ctx = LayerContext.from_config(
                index=i,
                config=self._config,
                checkpoint_path=checkpoint_path,
                device="cpu",  # Weight modification on CPU
                dtype=torch.float16,
            )
            contexts.append(ctx)

        logger.info("Built %d layer contexts", len(contexts))
        return contexts

    def _find_or_create_splitted_model(self) -> Path:
        """Locate or create the splitted model directory.

        Looks for an existing splitted_model directory under model_path.
        If not found, attempts to split using AirLLM's utilities. If that
        fails, looks for safetensors index file for direct loading.

        Returns:
            Path to the directory containing per-layer safetensors files.

        Raises:
            FileNotFoundError: If no splitted model or index can be found.
        """
        # Check for existing splitted_model directory
        splitted_path = self.model_path / "splitted_model"
        if splitted_path.exists() and any(splitted_path.glob("*.safetensors")):
            logger.info("Found existing splitted model at %s", splitted_path)
            return splitted_path

        # Check for alternative paths
        for name in ["splitted_model", "splitted_model.4bit", "splitted_model.8bit"]:
            candidate = self.model_path / name
            if candidate.exists() and any(candidate.glob("*.safetensors")):
                logger.info("Found splitted model at %s", candidate)
                return candidate

        # Attempt to split using AirLLM's utility
        try:
            from airllm.utils import find_or_create_local_splitted_path

            logger.info("Splitting model using AirLLM utilities...")
            _, saved_path = find_or_create_local_splitted_path(
                str(self.model_path)
            )
            logger.info("Model split complete: %s", saved_path)
            return Path(saved_path)
        except ImportError:
            logger.warning(
                "AirLLM not available for model splitting. "
                "Attempting direct safetensors loading."
            )
        except Exception as exc:
            logger.warning(
                "AirLLM model splitting failed: %s. "
                "Attempting direct safetensors loading.",
                exc,
            )

        # Fall back to looking for per-layer safetensors in model_path
        safetensors_files = list(self.model_path.glob("model.layers.*.safetensors"))
        if safetensors_files:
            logger.info(
                "Found %d per-layer safetensors files in %s",
                len(safetensors_files), self.model_path,
            )
            return self.model_path

        raise FileNotFoundError(
            f"No splitted model found at {self.model_path}. "
            f"Expected a 'splitted_model' directory with per-layer "
            f"safetensors files, or individual layer files in the model directory. "
            f"Run AirLLM model splitting first, or ensure the model is "
            f"already split into per-layer safetensors."
        )

    def _make_progress_update(
        self,
        operation_type: str,
        current_layer: int,
        layer_type: str,
        substep: str,
        substep_progress: float,
        run_start: float,
    ) -> ProgressUpdate:
        """Build a ProgressUpdate with current resource and timing data.

        Args:
            operation_type: Name of the current operation.
            current_layer: Current layer index.
            layer_type: "DeltaNet" or "Attention".
            substep: Current substep name.
            substep_progress: Substep completion (0.0 to 1.0).
            run_start: Timestamp when the run started.

        Returns:
            Populated ProgressUpdate dataclass.
        """
        snap = self._resource_monitor.get_snapshot()
        elapsed = time.time() - run_start
        lpm = self._profiler.layers_per_minute if self._profiler else 0.0

        # Estimate ETA
        eta: float | None = None
        if self._profiler and self._profiler.completed_layers > 0:
            eta = self._profiler.estimated_remaining(self._total_layers)
        elif lpm > 0:
            remaining = self._total_layers - current_layer
            eta = (remaining / lpm) * 60.0

        return ProgressUpdate(
            operation_type=operation_type,
            current_layer=current_layer,
            total_layers=self._total_layers,
            layer_type=layer_type,
            substep=substep,
            substep_progress=substep_progress,
            ram_used_gb=snap.ram_used_bytes / (1024 ** 3),
            ram_total_gb=snap.ram_total_bytes / (1024 ** 3),
            vram_used_gb=snap.vram_used_bytes / (1024 ** 3),
            vram_total_gb=snap.vram_total_bytes / (1024 ** 3),
            cpu_percent=snap.cpu_percent,
            elapsed_seconds=elapsed,
            eta_seconds=eta,
            layers_per_minute=lpm,
        )

    def _emit_progress(
        self,
        operation_type: str,
        current_layer: int,
        layer_type: str,
        substep: str,
        substep_progress: float,
        run_start: float,
    ) -> None:
        """Send a progress update to the callback if one is registered."""
        if self.progress_callback is None:
            return

        try:
            update = self._make_progress_update(
                operation_type=operation_type,
                current_layer=current_layer,
                layer_type=layer_type,
                substep=substep,
                substep_progress=substep_progress,
                run_start=run_start,
            )
            self.progress_callback(update)
        except Exception as exc:
            logger.debug("Progress callback error: %s", exc)

    def run(self) -> TrainerResult:
        """Main entry point. Iterates all layers and applies operations.

        This is the core training loop:
        1. Load model config
        2. Detect layer types
        3. Find/create splitted model
        4. For each layer: load -> apply ops -> validate -> save -> checkpoint
        5. Return summary

        Returns:
            TrainerResult with completed/skipped layers, warnings, and DLQ info.

        Raises:
            FileNotFoundError: If model or splitted model cannot be found.
            RuntimeError: If critical errors occur during processing.
        """
        run_start = time.time()
        result = TrainerResult()

        logger.info("=" * 60)
        logger.info("AEGIS AI Trainer — Starting run")
        logger.info("  Model: %s", self.model_path)
        logger.info("  Output: %s", self.output_path)
        logger.info("  Operations: %s", [type(op).__name__ for op in self.operations])
        logger.info("  Dry run: %s", self.dry_run)
        logger.info("=" * 60)

        # Step 1: Load model config
        self._load_model_config()

        # Step 2: Detect layer types (done in _load_model_config)
        logger.info(
            "Layer types: %d linear_attention (DeltaNet), %d full_attention",
            sum(1 for lt in self._layer_types if lt == 'linear_attention'),
            sum(1 for lt in self._layer_types if lt == 'full_attention'),
        )

        # Step 3: Find or create splitted model directory
        checkpoint_path = self._find_or_create_splitted_model()

        # Step 4: Build layer contexts
        layer_contexts = self._build_layer_contexts(checkpoint_path)

        # Record metadata in checkpoint
        self._checkpoint.set_metadata("model_path", str(self.model_path))
        self._checkpoint.set_metadata("output_path", str(self.output_path))
        self._checkpoint.set_metadata(
            "operations", [type(op).__name__ for op in self.operations]
        )
        self._checkpoint.set_metadata("total_layers", self._total_layers)
        self._checkpoint.set_metadata("start_time", run_start)

        # Step 5: Process each layer
        for layer_idx, ctx in enumerate(layer_contexts):
            layer_name = ctx.layer_name
            layer_type_str = (
                "DeltaNet" if ctx.is_deltanet else "Attention"
            )

            # 5a: Check checkpoint — skip if already completed
            if self._checkpoint.is_completed(layer_name):
                logger.info(
                    "Skipping layer %d/%d (%s) — already completed",
                    layer_idx + 1, self._total_layers, layer_name,
                )
                result.skipped_layers.append(layer_name)
                self._emit_progress(
                    "skip", layer_idx, layer_type_str,
                    "skipped", 1.0, run_start,
                )
                continue

            logger.info(
                "Processing layer %d/%d: %s [%s]",
                layer_idx + 1, self._total_layers, layer_name, layer_type_str,
            )

            if self._profiler:
                self._profiler.start_layer(layer_name)

            # 5b: Determine which operations should apply to this layer
            applicable_ops: list[LayerOperation] = []
            for op in self.operations:
                try:
                    if op.should_apply(ctx):
                        applicable_ops.append(op)
                except Exception as exc:
                    msg = f"should_apply() error: {exc}"
                    logger.warning(
                        "Layer %s, op %s: %s",
                        layer_name, type(op).__name__, msg,
                    )
                    result.warnings.append((ctx, op, msg))

            if not applicable_ops:
                logger.info(
                    "Layer %d/%d: no operations apply, skipping",
                    layer_idx + 1, self._total_layers,
                )
                # Still mark as completed (no work needed)
                if not self.dry_run:
                    self._checkpoint.mark_completed(layer_name)
                result.completed_layers.append(layer_name)

                if self._profiler:
                    self._profiler.end_layer(layer_name)

                self._emit_progress(
                    "none", layer_idx, layer_type_str,
                    "skipped", 1.0, run_start,
                )
                continue

            # 5c: Load layer weights from disk
            self._emit_progress(
                applicable_ops[0].name if hasattr(applicable_ops[0], 'name') else type(applicable_ops[0]).__name__,
                layer_idx, layer_type_str,
                "loading", 0.0, run_start,
            )

            if self._profiler:
                self._profiler.start_operation("load")

            try:
                state_dict = self._layer_io.load(ctx, device="cpu")
            except FileNotFoundError as exc:
                msg = f"Failed to load layer weights: {exc}"
                logger.error(msg)
                for op in applicable_ops:
                    self._queue.send_to_dlq(ctx, op, msg)
                result.warnings.append((ctx, None, msg))

                if self._profiler:
                    self._profiler.end_operation("load")
                    self._profiler.end_layer(layer_name)
                continue

            if self._profiler:
                self._profiler.end_operation("load")

            self._emit_progress(
                type(applicable_ops[0]).__name__,
                layer_idx, layer_type_str,
                "loading", 1.0, run_start,
            )

            # Notify weight visualizer with original (pre-modification) weights
            if self.weight_callback:
                try:
                    self.weight_callback(state_dict, ctx, "before")
                except Exception as exc:
                    logger.debug("Weight callback (before) error: %s", exc)

            # Keep original for validation (shallow copy of references)
            original_state_dict: dict[str, torch.Tensor] | None = None
            if self.enable_validation:
                original_state_dict = {k: v.clone() for k, v in state_dict.items()}

            # 5d: Execute each applicable operation via QQMS queue
            all_ops_ok = True
            for op_idx, op in enumerate(applicable_ops):
                op_name = type(op).__name__

                self._emit_progress(
                    op_name, layer_idx, layer_type_str,
                    "modifying",
                    op_idx / len(applicable_ops),
                    run_start,
                )

                if self._profiler:
                    self._profiler.start_operation(op_name)

                # Build queue item
                queue_item = QueueItem(
                    operation=op,
                    state_dict=state_dict,
                    context=ctx,
                    priority=float(op_idx),  # Earlier ops get higher priority
                    max_attempts=self._queue._config.max_attempts,
                    age_boost=self._queue._config.age_boost,
                )

                try:
                    state_dict = self._queue.execute(queue_item)
                except RuntimeError as exc:
                    # Operation exhausted retries and went to DLQ
                    msg = str(exc)
                    logger.error(
                        "Layer %s, op %s failed permanently: %s",
                        layer_name, op_name, msg,
                    )
                    result.warnings.append((ctx, op, msg))
                    all_ops_ok = False

                    if self._profiler:
                        self._profiler.end_operation(op_name)
                    continue
                except TimeoutError as exc:
                    # Resource throttle timeout
                    msg = f"Resource throttle timeout: {exc}"
                    logger.error(
                        "Layer %s, op %s throttle timeout: %s",
                        layer_name, op_name, msg,
                    )
                    self._queue.send_to_dlq(ctx, op, msg)
                    result.warnings.append((ctx, op, msg))
                    all_ops_ok = False

                    if self._profiler:
                        self._profiler.end_operation(op_name)
                    continue

                if self._profiler:
                    self._profiler.end_operation(op_name)

                # 5e: Validate if enabled
                if self.enable_validation and original_state_dict is not None:
                    self._emit_progress(
                        op_name, layer_idx, layer_type_str,
                        "verifying",
                        (op_idx + 0.5) / len(applicable_ops),
                        run_start,
                    )

                    if self._profiler:
                        self._profiler.start_operation(f"validate_{op_name}")

                    try:
                        valid = op.validate(original_state_dict, state_dict, ctx)
                        if not valid:
                            msg = (
                                f"Validation failed for {op_name} on {layer_name}. "
                                f"NaN or Inf detected in modified weights."
                            )
                            logger.error(msg)
                            result.warnings.append((ctx, op, msg))
                            self._queue.send_to_dlq(ctx, op, msg)
                            all_ops_ok = False
                    except Exception as exc:
                        msg = f"Validation error: {exc}"
                        logger.warning(
                            "Layer %s, op %s validation error: %s",
                            layer_name, op_name, exc,
                        )
                        result.warnings.append((ctx, op, msg))

                    if self._profiler:
                        self._profiler.end_operation(f"validate_{op_name}")

            # Notify weight visualizer with modified (post-operation) weights
            if self.weight_callback:
                try:
                    self.weight_callback(state_dict, ctx, "after")
                except Exception as exc:
                    logger.debug("Weight callback (after) error: %s", exc)

            # 5f: Save modified weights to output_path
            if all_ops_ok and not self.dry_run:
                self._emit_progress(
                    type(applicable_ops[-1]).__name__,
                    layer_idx, layer_type_str,
                    "saving", 0.5, run_start,
                )

                if self._profiler:
                    self._profiler.start_operation("save")

                try:
                    self._layer_io.save(
                        state_dict, ctx, output_path=self.output_path,
                    )
                except Exception as exc:
                    msg = f"Failed to save layer: {exc}"
                    logger.error(
                        "Layer %s save failed: %s", layer_name, exc,
                    )
                    result.warnings.append((ctx, None, msg))
                    for op in applicable_ops:
                        self._queue.send_to_dlq(ctx, op, msg)

                    if self._profiler:
                        self._profiler.end_operation("save")
                        self._profiler.end_layer(layer_name)

                    # Free memory before continuing
                    del state_dict
                    if original_state_dict is not None:
                        del original_state_dict
                    _clean_memory()
                    continue

                if self._profiler:
                    self._profiler.end_operation("save")
            elif self.dry_run:
                logger.info(
                    "DRY RUN: would save layer %s (%d ops applied)",
                    layer_name, len(applicable_ops),
                )

            # 5g: Mark layer completed in checkpoint
            if all_ops_ok and not self.dry_run:
                self._checkpoint.mark_completed(layer_name)

            result.completed_layers.append(layer_name)

            if self._profiler:
                self._profiler.end_layer(layer_name)

            # 5h: Free memory — never hold more than one layer
            del state_dict
            if original_state_dict is not None:
                del original_state_dict
            gc.collect()
            _clean_memory()

            # 5i: Emit final progress for this layer
            self._emit_progress(
                type(applicable_ops[-1]).__name__,
                layer_idx, layer_type_str,
                "saving", 1.0, run_start,
            )

            logger.info(
                "Completed layer %d/%d: %s [%s] (%d ops applied)",
                layer_idx + 1, self._total_layers,
                layer_name, layer_type_str,
                len(applicable_ops),
            )

        # Step 6: Build and return result
        result.total_time = time.time() - run_start
        result.dlq_entries = self._queue.get_dlq_report()

        # Log profiling summary
        if self._profiler:
            logger.info("Profiling: %s", self._profiler.get_summary())

        # Log queue stats
        stats = self._queue.get_stats()
        logger.info(stats.summary())

        logger.info(result.summary())

        return result

    def run_single_layer(
        self,
        layer_name: str,
        operations: list[LayerOperation] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Process a single layer (for testing/debugging).

        Loads the specified layer, applies operations, and returns the
        modified state dict without saving to disk or updating checkpoints.

        Args:
            layer_name: Layer name (e.g. "model.layers.0.").
            operations: Operations to apply. Defaults to self.operations.

        Returns:
            Modified state dict for the layer.

        Raises:
            FileNotFoundError: If model config or layer file cannot be found.
            ValueError: If the layer name is invalid.
        """
        ops = operations if operations is not None else self.operations

        # Load config if not already loaded
        if self._config is None:
            self._load_model_config()

        # Find splitted model
        checkpoint_path = self._find_or_create_splitted_model()

        # Parse layer index from name
        try:
            # Expected format: "model.layers.N."
            parts = layer_name.rstrip('.').split('.')
            layer_idx = int(parts[-1])
        except (ValueError, IndexError):
            raise ValueError(
                f"Cannot parse layer index from '{layer_name}'. "
                f"Expected format: 'model.layers.N.'"
            )

        # Build context
        ctx = LayerContext.from_config(
            index=layer_idx,
            config=self._config,
            checkpoint_path=checkpoint_path,
            device="cpu",
            dtype=torch.float16,
        )

        # Load weights
        logger.info("Loading single layer: %s", layer_name)
        state_dict = self._layer_io.load(ctx, device="cpu")

        # Apply operations
        for op in ops:
            op_name = type(op).__name__
            if not op.should_apply(ctx):
                logger.info(
                    "Skipping %s on %s (should_apply=False)",
                    op_name, layer_name,
                )
                continue

            logger.info("Applying %s to %s", op_name, layer_name)
            queue_item = QueueItem(
                operation=op,
                state_dict=state_dict,
                context=ctx,
                priority=0.0,
            )
            state_dict = self._queue.execute(queue_item)

        return state_dict

    @property
    def profiler(self) -> TrainerProfiler | None:
        """Access the profiler (None if profiling is disabled)."""
        return self._profiler

    @property
    def queue(self) -> QQMSQueue:
        """Access the QQMS queue."""
        return self._queue

    @property
    def checkpoint(self) -> CheckpointManager:
        """Access the checkpoint manager."""
        return self._checkpoint

    @property
    def resource_monitor(self) -> ResourceMonitor:
        """Access the resource monitor."""
        return self._resource_monitor

    def __repr__(self) -> str:
        return (
            f"LayerTrainer("
            f"model={self.model_path}, "
            f"output={self.output_path}, "
            f"ops={len(self.operations)}, "
            f"dry_run={self.dry_run})"
        )
