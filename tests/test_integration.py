"""
Integration tests for AEGIS AI Trainer.

Verifies that core components work together correctly: LayerContext,
all 6 operations, ResourceMonitor, CheckpointManager, TrainerProfiler,
QueueItem, DeadLetterQueue, QQMSConfig, and CLI help output.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

# ── Core imports ────────────────────────────────────────────────
from aegis_trainer.layer_context import (
    LayerContext,
    _get_layer_type,
    _LAYER_TYPES,
    _QWEN3_NEXT_TOTAL_LAYERS,
)
from aegis_trainer.ops.abliteration import AbliterationOp
from aegis_trainer.ops.longrope import LongRoPEOp
from aegis_trainer.ops.weight_inspect import WeightInspectOp
from aegis_trainer.ops.lora_merge import LoRAMergeOp
from aegis_trainer.ops.expert_prune import ExpertPruneOp
from aegis_trainer.ops.quantize import QuantizeOp
from aegis_trainer.utils.resource_monitor import (
    ResourceLimits,
    ResourceMonitor,
    ResourceSnapshot,
)
from aegis_trainer.utils.checkpoint import CheckpointManager
from aegis_trainer.utils.profiler import TrainerProfiler
from aegis_trainer.queue.queue_item import QueueItem
from aegis_trainer.queue.dlq import DeadLetterQueue, DLQEntry
from aegis_trainer.queue.qqms import QQMSConfig, QQMSQueue, QQMSStats


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_linear_ctx(index: int = 0) -> LayerContext:
    """Create a LayerContext for a DeltaNet (linear_attention) layer."""
    return LayerContext(
        layer_index=index,
        layer_name=f"model.layers.{index}.",
        layer_type="linear_attention",
        total_layers=48,
        device="cpu",
        dtype=torch.float16,
    )


def _make_full_ctx(index: int = 3) -> LayerContext:
    """Create a LayerContext for a full_attention layer."""
    return LayerContext(
        layer_index=index,
        layer_name=f"model.layers.{index}.",
        layer_type="full_attention",
        total_layers=48,
        device="cpu",
        dtype=torch.float16,
    )


# ═══════════════════════════════════════════════════════════════
# a) test_layer_context_creation
# ═══════════════════════════════════════════════════════════════

class TestLayerContextCreation:
    """Verify manual LayerContext construction and all computed properties."""

    def test_basic_properties(self):
        ctx = _make_linear_ctx(0)
        assert ctx.layer_index == 0
        assert ctx.layer_name == "model.layers.0."
        assert ctx.layer_type == "linear_attention"
        assert ctx.total_layers == 48

    def test_is_deltanet_true_for_linear(self):
        ctx = _make_linear_ctx(0)
        assert ctx.is_deltanet is True

    def test_is_deltanet_false_for_full(self):
        ctx = _make_full_ctx(3)
        assert ctx.is_deltanet is False

    def test_is_rope_enabled(self):
        assert _make_linear_ctx(0).is_rope_enabled is False
        assert _make_full_ctx(3).is_rope_enabled is True

    def test_safetensors_filename(self):
        ctx = _make_linear_ctx(5)
        assert ctx.safetensors_filename == "model.layers.5.safetensors"

    def test_safetensors_path(self):
        ctx = LayerContext(
            layer_index=5,
            layer_name="model.layers.5.",
            layer_type="linear_attention",
            checkpoint_path=Path("/tmp/splitted_model"),
        )
        assert ctx.safetensors_path == Path("/tmp/splitted_model/model.layers.5.safetensors")

    def test_done_marker_path(self):
        ctx = LayerContext(
            layer_index=5,
            layer_name="model.layers.5.",
            layer_type="linear_attention",
            checkpoint_path=Path("/tmp/splitted_model"),
        )
        assert ctx.done_marker_path == Path(
            "/tmp/splitted_model/model.layers.5.safetensors.done"
        )

    def test_layer_fraction(self):
        ctx_first = _make_linear_ctx(0)
        ctx_last = LayerContext(
            layer_index=47,
            layer_name="model.layers.47.",
            layer_type="full_attention",
            total_layers=48,
        )
        assert ctx_first.layer_fraction == pytest.approx(0.0)
        assert ctx_last.layer_fraction == pytest.approx(1.0)

    def test_frozen(self):
        ctx = _make_linear_ctx(0)
        with pytest.raises(AttributeError):
            ctx.layer_index = 5  # type: ignore

    def test_defaults_match_qwen3_next(self):
        ctx = _make_linear_ctx(0)
        assert ctx.num_experts == 512
        assert ctx.num_active_experts == 10
        assert ctx.hidden_size == 2048
        assert ctx.head_dim == 256
        assert ctx.num_attention_heads == 16
        assert ctx.num_kv_heads == 2


# ═══════════════════════════════════════════════════════════════
# b) test_layer_context_patterns
# ═══════════════════════════════════════════════════════════════

class TestLayerContextPatterns:
    """Verify Qwen3-Next layer type patterns."""

    def test_total_layer_count(self):
        assert len(_LAYER_TYPES) == 48

    def test_linear_attention_count(self):
        linear_count = sum(1 for lt in _LAYER_TYPES if lt == "linear_attention")
        assert linear_count == 36

    def test_full_attention_count(self):
        full_count = sum(1 for lt in _LAYER_TYPES if lt == "full_attention")
        assert full_count == 12

    def test_repeating_pattern(self):
        """Verify pattern: 3 linear + 1 full, repeating 12 times."""
        for cycle in range(12):
            base = cycle * 4
            assert _LAYER_TYPES[base] == "linear_attention"
            assert _LAYER_TYPES[base + 1] == "linear_attention"
            assert _LAYER_TYPES[base + 2] == "linear_attention"
            assert _LAYER_TYPES[base + 3] == "full_attention"

    def test_get_layer_type_without_config(self):
        assert _get_layer_type(0) == "linear_attention"
        assert _get_layer_type(1) == "linear_attention"
        assert _get_layer_type(2) == "linear_attention"
        assert _get_layer_type(3) == "full_attention"
        assert _get_layer_type(7) == "full_attention"

    def test_get_layer_type_beyond_range_cycles(self):
        # Beyond 48 layers, should cycle the pattern
        assert _get_layer_type(48) == "linear_attention"
        assert _get_layer_type(51) == "full_attention"

    def test_full_attention_indices(self):
        """Full attention layers should be at indices 3, 7, 11, ..., 47."""
        expected = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]
        actual = [i for i, lt in enumerate(_LAYER_TYPES) if lt == "full_attention"]
        assert actual == expected


# ═══════════════════════════════════════════════════════════════
# c) test_operation_construction
# ═══════════════════════════════════════════════════════════════

class TestOperationConstruction:
    """Instantiate each of the 6 operations with valid config."""

    def test_abliteration_op(self):
        directions = torch.randn(1, 2048)
        op = AbliterationOp(refusal_directions=directions)
        assert op.name == "abliterate"

    def test_abliteration_op_1d_direction(self):
        direction = torch.randn(2048)
        op = AbliterationOp(refusal_directions=direction)
        assert op.refusal_directions.shape == (1, 2048)

    def test_longrope_op(self):
        factors = torch.ones(128)  # head_dim=256, so 128 = 256//2
        op = LongRoPEOp(rescale_factors=factors)
        assert op.name == "longrope"
        assert op.target_max_position_embeddings == 524288

    def test_weight_inspect_op(self):
        op = WeightInspectOp()
        assert op.name == "weight_inspect"

    def test_lora_merge_op(self):
        lora_state_dict = {}
        op = LoRAMergeOp(lora_state_dict=lora_state_dict)
        assert op.name == "lora_merge"

    def test_expert_prune_op(self):
        op = ExpertPruneOp(experts_to_prune=[0, 1, 2])
        assert op.name == "expert_prune"

    def test_expert_prune_op_invalid_mode(self):
        with pytest.raises(ValueError, match="Invalid prune_mode"):
            ExpertPruneOp(prune_mode="invalid")

    def test_quantize_op(self):
        op = QuantizeOp(target_dtype=torch.float16)
        assert op.name == "quantize"

    def test_quantize_op_invalid_dtype(self):
        with pytest.raises(ValueError, match="Unsupported target_dtype"):
            QuantizeOp(target_dtype=torch.int8)


# ═══════════════════════════════════════════════════════════════
# d) test_operation_should_apply
# ═══════════════════════════════════════════════════════════════

class TestOperationShouldApply:
    """Verify should_apply() returns correct True/False per layer type."""

    def setup_method(self):
        self.linear_ctx = _make_linear_ctx(0)
        self.full_ctx = _make_full_ctx(3)

    def test_abliteration_applies_to_all(self):
        directions = torch.randn(1, 2048)
        op = AbliterationOp(refusal_directions=directions)
        assert op.should_apply(self.linear_ctx) is True
        assert op.should_apply(self.full_ctx) is True

    def test_longrope_only_full_attention(self):
        factors = torch.ones(128)
        op = LongRoPEOp(rescale_factors=factors)
        assert op.should_apply(self.linear_ctx) is False
        assert op.should_apply(self.full_ctx) is True

    def test_weight_inspect_applies_to_all(self):
        op = WeightInspectOp()
        assert op.should_apply(self.linear_ctx) is True
        assert op.should_apply(self.full_ctx) is True

    def test_lora_merge_based_on_adapter_weights(self):
        # Empty adapter: should not apply to any layer
        op_empty = LoRAMergeOp(lora_state_dict={})
        assert op_empty.should_apply(self.linear_ctx) is False
        assert op_empty.should_apply(self.full_ctx) is False

        # Adapter with weights for layer 0
        lora_dict = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, 2048),
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(2048, 16),
        }
        op_with = LoRAMergeOp(lora_state_dict=lora_dict)
        assert op_with.should_apply(self.linear_ctx) is True  # layer 0
        assert op_with.should_apply(self.full_ctx) is False   # layer 3

    def test_expert_prune_based_on_config(self):
        # No experts to prune: should not apply
        op_empty = ExpertPruneOp(experts_to_prune=[])
        assert op_empty.should_apply(self.linear_ctx) is False

        # With experts to prune and num_experts > 0 (default 512)
        op_with = ExpertPruneOp(experts_to_prune=[0, 1])
        assert op_with.should_apply(self.linear_ctx) is True
        assert op_with.should_apply(self.full_ctx) is True

        # Layer with no experts
        no_expert_ctx = LayerContext(
            layer_index=0,
            layer_name="model.layers.0.",
            layer_type="linear_attention",
            num_experts=0,
        )
        assert op_with.should_apply(no_expert_ctx) is False

    def test_quantize_applies_to_all(self):
        op = QuantizeOp()
        assert op.should_apply(self.linear_ctx) is True
        assert op.should_apply(self.full_ctx) is True


# ═══════════════════════════════════════════════════════════════
# e) test_resource_monitor
# ═══════════════════════════════════════════════════════════════

class TestResourceMonitor:
    """Verify ResourceMonitor instantiation and snapshot capture."""

    def test_default_limits(self):
        limits = ResourceLimits()
        assert limits.max_cpu_percent == 85.0
        assert limits.max_ram_percent == 90.0
        assert limits.max_vram_percent == 85.0

    def test_snapshot_returns_valid_data(self):
        monitor = ResourceMonitor()
        snap = monitor.get_snapshot()
        assert isinstance(snap, ResourceSnapshot)
        assert snap.cpu_percent >= 0.0
        assert snap.ram_used_bytes > 0
        assert snap.ram_total_bytes > 0
        assert snap.timestamp > 0

    def test_snapshot_ram_percent(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_used_bytes=60 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=0,
            vram_total_bytes=0,
            timestamp=time.time(),
        )
        assert snap.ram_percent == pytest.approx(50.0)

    def test_snapshot_vram_percent_zero_total(self):
        snap = ResourceSnapshot(
            cpu_percent=50.0,
            ram_used_bytes=60 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=0,
            vram_total_bytes=0,
            timestamp=time.time(),
        )
        assert snap.vram_percent == 0.0

    def test_resource_monitor_with_custom_limits(self):
        limits = ResourceLimits(max_cpu_percent=50.0, max_ram_percent=60.0)
        monitor = ResourceMonitor(limits=limits)
        assert monitor.limits.max_cpu_percent == 50.0
        assert monitor.limits.max_ram_percent == 60.0


# ═══════════════════════════════════════════════════════════════
# f) test_checkpoint_manager
# ═══════════════════════════════════════════════════════════════

class TestCheckpointManager:
    """Verify CheckpointManager save/load/clear persistence."""

    def test_mark_and_check_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(output_path=tmpdir)
            assert ckpt.is_completed("model.layers.0.") is False
            ckpt.mark_completed("model.layers.0.")
            assert ckpt.is_completed("model.layers.0.") is True

    def test_persistence_across_instances(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt1 = CheckpointManager(output_path=tmpdir)
            ckpt1.mark_completed("model.layers.0.")
            ckpt1.mark_completed("model.layers.1.")

            # New instance from same path should restore
            ckpt2 = CheckpointManager(output_path=tmpdir)
            assert ckpt2.is_completed("model.layers.0.") is True
            assert ckpt2.is_completed("model.layers.1.") is True
            assert ckpt2.num_completed == 2

    def test_reset_clears_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(output_path=tmpdir)
            ckpt.mark_completed("model.layers.0.")
            assert ckpt.num_completed == 1
            ckpt.reset()
            assert ckpt.num_completed == 0

            # After reset and reload, should be empty
            ckpt2 = CheckpointManager(output_path=tmpdir)
            assert ckpt2.num_completed == 0

    def test_metadata_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(output_path=tmpdir)
            ckpt.set_metadata("model_path", "/models/qwen3")
            ckpt.set_metadata("total_layers", 48)

            # Reload and verify
            ckpt2 = CheckpointManager(output_path=tmpdir)
            assert ckpt2.get_metadata("model_path") == "/models/qwen3"
            assert ckpt2.get_metadata("total_layers") == 48

    def test_get_completed_sorted(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = CheckpointManager(output_path=tmpdir)
            ckpt.mark_completed("model.layers.2.")
            ckpt.mark_completed("model.layers.0.")
            ckpt.mark_completed("model.layers.1.")
            result = ckpt.get_completed()
            assert result == [
                "model.layers.0.",
                "model.layers.1.",
                "model.layers.2.",
            ]


# ═══════════════════════════════════════════════════════════════
# g) test_profiler
# ═══════════════════════════════════════════════════════════════

class TestTrainerProfiler:
    """Verify TrainerProfiler timing and report generation."""

    def test_basic_layer_timing(self):
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        time.sleep(0.01)
        elapsed = profiler.end_layer("model.layers.0.")
        assert elapsed > 0.0
        assert profiler.completed_layers == 1

    def test_operation_timing(self):
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        profiler.start_operation("load")
        time.sleep(0.01)
        op_elapsed = profiler.end_operation("load")
        assert op_elapsed > 0.0
        profiler.end_layer("model.layers.0.")

    def test_report_generation(self):
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        profiler.start_operation("abliteration")
        time.sleep(0.01)
        profiler.end_operation("abliteration")
        profiler.end_layer("model.layers.0.")

        report = profiler.get_report()
        assert report["completed_layers"] == 1
        assert "model.layers.0." in report["per_layer"]
        assert "abliteration" in report["per_operation"]
        assert report["per_operation"]["abliteration"]["count"] == 1

    def test_summary_string(self):
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        profiler.end_layer("model.layers.0.")
        summary = profiler.get_summary()
        assert "1 layers done" in summary

    def test_estimated_remaining(self):
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        time.sleep(0.01)
        profiler.end_layer("model.layers.0.")
        remaining = profiler.estimated_remaining(48)
        assert remaining > 0.0

    def test_end_layer_without_start_raises(self):
        profiler = TrainerProfiler()
        with pytest.raises(KeyError):
            profiler.end_layer("model.layers.0.")

    def test_end_operation_without_start_raises(self):
        profiler = TrainerProfiler()
        with pytest.raises(KeyError):
            profiler.end_operation("load")


# ═══════════════════════════════════════════════════════════════
# h) test_queue_item_priority
# ═══════════════════════════════════════════════════════════════

class TestQueueItemPriority:
    """Verify QueueItem effective_priority aging."""

    def test_effective_priority_decreases_with_age(self):
        mock_op = MagicMock()
        mock_ctx = MagicMock()
        item = QueueItem(
            operation=mock_op,
            state_dict={},
            context=mock_ctx,
            priority=10.0,
            age_boost=1.0,  # 1.0 per second for fast testing
        )
        initial = item.effective_priority
        time.sleep(0.1)
        later = item.effective_priority
        assert later < initial

    def test_higher_base_priority_is_less_urgent(self):
        mock_op = MagicMock()
        mock_ctx = MagicMock()
        now = time.time()
        item_urgent = QueueItem(
            operation=mock_op,
            state_dict={},
            context=mock_ctx,
            priority=1.0,
            created_at=now,
            age_boost=0.0,
        )
        item_low = QueueItem(
            operation=mock_op,
            state_dict={},
            context=mock_ctx,
            priority=10.0,
            created_at=now,
            age_boost=0.0,
        )
        assert item_urgent.effective_priority < item_low.effective_priority

    def test_is_exhausted(self):
        mock_op = MagicMock()
        mock_ctx = MagicMock()
        item = QueueItem(
            operation=mock_op,
            state_dict={},
            context=mock_ctx,
            priority=1.0,
            max_attempts=3,
        )
        assert item.is_exhausted is False
        item.increment_attempt()
        item.increment_attempt()
        item.increment_attempt()
        assert item.is_exhausted is True

    def test_queue_item_ordering(self):
        mock_op = MagicMock()
        mock_ctx = MagicMock()
        now = time.time()
        item_a = QueueItem(
            operation=mock_op, state_dict={}, context=mock_ctx,
            priority=1.0, created_at=now, age_boost=0.0,
        )
        item_b = QueueItem(
            operation=mock_op, state_dict={}, context=mock_ctx,
            priority=5.0, created_at=now, age_boost=0.0,
        )
        assert item_a < item_b  # lower priority value = more urgent


# ═══════════════════════════════════════════════════════════════
# i) test_dlq
# ═══════════════════════════════════════════════════════════════

class TestDeadLetterQueue:
    """Verify DeadLetterQueue add, retrieve, clear operations."""

    def test_add_and_retrieve(self):
        dlq = DeadLetterQueue(max_size=10)
        mock_item = MagicMock()
        mock_item.operation = MagicMock()
        type(mock_item.operation).__name__ = "AbliterationOp"
        mock_item.context = _make_linear_ctx(0)
        mock_item.attempt_count = 3

        dlq.add(mock_item, reason="Test failure")
        assert dlq.size == 1

        report = dlq.get_report()
        assert len(report) == 1
        assert report[0].reason == "Test failure"
        assert report[0].operation_name == "AbliterationOp"

    def test_add_from_context(self):
        dlq = DeadLetterQueue(max_size=10)
        ctx = _make_linear_ctx(5)
        op = WeightInspectOp()

        dlq.add_from_context(ctx, op, "Direct add test")
        assert dlq.size == 1
        report = dlq.get_report()
        assert report[0].layer_name == "model.layers.5."

    def test_clear(self):
        dlq = DeadLetterQueue(max_size=10)
        ctx = _make_linear_ctx(0)
        op = WeightInspectOp()
        dlq.add_from_context(ctx, op, "entry 1")
        dlq.add_from_context(ctx, op, "entry 2")
        assert dlq.size == 2
        cleared = dlq.clear()
        assert cleared == 2
        assert dlq.size == 0

    def test_get_summary(self):
        dlq = DeadLetterQueue(max_size=10)
        ctx = _make_linear_ctx(0)
        dlq.add_from_context(ctx, WeightInspectOp(), "fail 1")
        dlq.add_from_context(ctx, QuantizeOp(), "fail 2")

        summary = dlq.get_summary()
        assert summary["total"] == 2
        assert "WeightInspectOp" in summary["by_operation"]
        assert "QuantizeOp" in summary["by_operation"]

    def test_len_and_bool(self):
        dlq = DeadLetterQueue(max_size=10)
        assert len(dlq) == 0
        assert not dlq

        ctx = _make_linear_ctx(0)
        dlq.add_from_context(ctx, WeightInspectOp(), "test")
        assert len(dlq) == 1
        assert dlq


# ═══════════════════════════════════════════════════════════════
# j) test_qqms_config
# ═══════════════════════════════════════════════════════════════

class TestQQMSConfig:
    """Verify QQMSConfig defaults and custom params."""

    def test_defaults(self):
        config = QQMSConfig()
        assert config.max_queue_size == 256
        assert config.max_attempts == 3
        assert config.age_boost == 0.01
        assert config.overflow_strategy == "drop_lowest"
        assert config.throttle_check_interval == 5.0
        assert config.max_throttle_wait == 300.0
        assert config.spill_dir is None

    def test_custom_values(self):
        config = QQMSConfig(
            max_queue_size=128,
            max_attempts=5,
            age_boost=0.05,
            overflow_strategy="block",
            throttle_check_interval=2.0,
            max_throttle_wait=60.0,
            spill_dir="/tmp/spill",
        )
        assert config.max_queue_size == 128
        assert config.max_attempts == 5
        assert config.overflow_strategy == "block"

    def test_qqms_queue_creation(self):
        config = QQMSConfig()
        queue = QQMSQueue(config=config)
        assert queue.pending_count == 0

    def test_qqms_stats_defaults(self):
        stats = QQMSStats()
        assert stats.items_processed == 0
        assert stats.items_failed == 0
        assert stats.avg_time_per_item == 0.0
        assert stats.success_rate == 1.0

    def test_qqms_stats_recording(self):
        stats = QQMSStats()
        stats.record_success(1.5)
        stats.record_success(2.5)
        assert stats.items_processed == 2
        assert stats.avg_time_per_item == pytest.approx(2.0)
        assert stats.success_rate == 1.0

        stats.record_dlq()
        assert stats.items_dlq == 1
        # success_rate = 2 / (2 + 1) = 0.667
        assert stats.success_rate == pytest.approx(2.0 / 3.0)


# ═══════════════════════════════════════════════════════════════
# k) test_cli_help
# ═══════════════════════════════════════════════════════════════

class TestCLIHelp:
    """Verify CLI help output for main command and subcommands."""

    def test_main_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AEGIS AI Trainer" in result.output

    def test_run_subcommand_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run a training/modification operation" in result.output

    def test_inspect_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_status_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "system resources" in result.output.lower() or "status" in result.output.lower()

    def test_queue_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["queue", "--help"])
        assert result.exit_code == 0
        assert "queue" in result.output.lower()

    def test_version_flag(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_run_abliterate_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "abliterate", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_run_longrope_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "longrope", "--help"])
        assert result.exit_code == 0
        assert "--target-context" in result.output

    def test_run_lora_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "lora", "--help"])
        assert result.exit_code == 0
        assert "--adapter" in result.output

    def test_run_quantize_help(self):
        from click.testing import CliRunner
        from aegis_trainer.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["run", "quantize", "--help"])
        assert result.exit_code == 0
        assert "--quant-type" in result.output
