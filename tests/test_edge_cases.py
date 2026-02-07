"""
Edge-case tests for AEGIS AI Trainer.

Covers boundary conditions and error-recovery paths:
  - ResourceMonitor over-limit detection
  - DeadLetterQueue at max capacity
  - OverflowManager strategies (DROP_LOWEST, BLOCK)
  - CheckpointManager with corrupted JSON
  - LayerContext for edge layer indices (0, 47)
  - Operation apply() with mock tensors
  - QQMSQueue execute() retry and DLQ routing

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from aegis_trainer.layer_context import LayerContext, _LAYER_TYPES
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
from aegis_trainer.queue.overflow import OverflowManager, OverflowStrategy
from aegis_trainer.queue.qqms import QQMSConfig, QQMSQueue, QQMSStats


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _make_ctx(index: int = 0, layer_type: str = "linear_attention") -> LayerContext:
    return LayerContext(
        layer_index=index,
        layer_name=f"model.layers.{index}.",
        layer_type=layer_type,
        total_layers=48,
        device="cpu",
        dtype=torch.float16,
    )


def _make_queue_item(priority: float = 1.0, max_attempts: int = 3) -> QueueItem:
    mock_op = MagicMock()
    mock_op.apply = MagicMock(return_value={"w": torch.randn(4, 4)})
    type(mock_op).__name__ = "MockOp"
    return QueueItem(
        operation=mock_op,
        state_dict={"w": torch.randn(4, 4)},
        context=_make_ctx(0),
        priority=priority,
        max_attempts=max_attempts,
        age_boost=0.0,
    )


# ═══════════════════════════════════════════════════════════════
# ResourceMonitor over-limit detection
# ═══════════════════════════════════════════════════════════════

class TestResourceMonitorOverLimit:
    """Test that ResourceMonitor correctly detects over-limit conditions."""

    def test_high_cpu_triggers_over_threshold(self):
        limits = ResourceLimits(max_cpu_percent=50.0)
        monitor = ResourceMonitor(limits=limits)
        # Mock get_snapshot to return high CPU
        fake_snap = ResourceSnapshot(
            cpu_percent=90.0,
            ram_used_bytes=10 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=0,
            vram_total_bytes=0,
            timestamp=time.time(),
        )
        with patch.object(monitor, 'get_snapshot', return_value=fake_snap):
            assert monitor.is_over_threshold() is True

    def test_high_ram_percent_triggers_over_threshold(self):
        limits = ResourceLimits(max_ram_percent=60.0)
        monitor = ResourceMonitor(limits=limits)
        fake_snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_used_bytes=100 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=0,
            vram_total_bytes=0,
            timestamp=time.time(),
        )
        with patch.object(monitor, 'get_snapshot', return_value=fake_snap):
            assert monitor.is_over_threshold() is True

    def test_high_ram_bytes_triggers_over_threshold(self):
        limits = ResourceLimits(max_ram_bytes=50 * 1024**3)
        monitor = ResourceMonitor(limits=limits)
        fake_snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_used_bytes=60 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=0,
            vram_total_bytes=0,
            timestamp=time.time(),
        )
        with patch.object(monitor, 'get_snapshot', return_value=fake_snap):
            assert monitor.is_over_threshold() is True

    def test_high_vram_triggers_over_threshold(self):
        limits = ResourceLimits(max_vram_percent=80.0)
        monitor = ResourceMonitor(limits=limits)
        fake_snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_used_bytes=10 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=10 * 1024**3,
            vram_total_bytes=11 * 1024**3,
            timestamp=time.time(),
        )
        with patch.object(monitor, 'get_snapshot', return_value=fake_snap):
            assert monitor.is_over_threshold() is True

    def test_under_all_limits(self):
        limits = ResourceLimits(
            max_cpu_percent=99.0,
            max_ram_percent=99.0,
            max_ram_bytes=200 * 1024**3,
            max_vram_percent=99.0,
            max_vram_bytes=20 * 1024**3,
        )
        monitor = ResourceMonitor(limits=limits)
        fake_snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_used_bytes=10 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=1 * 1024**3,
            vram_total_bytes=11 * 1024**3,
            timestamp=time.time(),
        )
        with patch.object(monitor, 'get_snapshot', return_value=fake_snap):
            assert monitor.is_over_threshold() is False

    def test_vram_check_skipped_when_no_vram(self):
        """When vram_total_bytes is 0, VRAM checks should be skipped."""
        limits = ResourceLimits(max_vram_percent=10.0)  # Very low threshold
        monitor = ResourceMonitor(limits=limits)
        fake_snap = ResourceSnapshot(
            cpu_percent=10.0,
            ram_used_bytes=10 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=0,
            vram_total_bytes=0,
            timestamp=time.time(),
        )
        with patch.object(monitor, 'get_snapshot', return_value=fake_snap):
            # Should not trigger even though threshold is low, since no VRAM
            assert monitor.is_over_threshold() is False

    def test_check_and_throttle_timeout(self):
        """Throttle should raise TimeoutError if resources stay high."""
        limits = ResourceLimits(max_cpu_percent=50.0)
        monitor = ResourceMonitor(
            limits=limits,
            throttle_sleep_seconds=0.01,
            throttle_max_wait_seconds=0.05,
        )
        fake_snap = ResourceSnapshot(
            cpu_percent=90.0,
            ram_used_bytes=10 * 1024**3,
            ram_total_bytes=120 * 1024**3,
            vram_used_bytes=0,
            vram_total_bytes=0,
            timestamp=time.time(),
        )
        with patch.object(monitor, 'get_snapshot', return_value=fake_snap):
            with pytest.raises(TimeoutError):
                monitor.check_and_throttle()


# ═══════════════════════════════════════════════════════════════
# DLQ at max capacity
# ═══════════════════════════════════════════════════════════════

class TestDLQMaxCapacity:
    """Test DeadLetterQueue behavior when at max capacity."""

    def test_max_size_eviction(self):
        """When DLQ is full, oldest entries are silently dropped (FIFO)."""
        dlq = DeadLetterQueue(max_size=3)
        ctx = _make_ctx(0)
        op = WeightInspectOp()

        dlq.add_from_context(ctx, op, "entry 1")
        dlq.add_from_context(ctx, op, "entry 2")
        dlq.add_from_context(ctx, op, "entry 3")
        assert dlq.size == 3

        # Adding a 4th should evict the oldest
        dlq.add_from_context(ctx, op, "entry 4")
        assert dlq.size == 3

        report = dlq.get_report()
        reasons = [e.reason for e in report]
        assert "entry 1" not in reasons  # Oldest was evicted
        assert "entry 4" in reasons

    def test_max_size_one(self):
        """DLQ with max_size=1 only retains the latest entry."""
        dlq = DeadLetterQueue(max_size=1)
        ctx = _make_ctx(0)
        op = WeightInspectOp()

        dlq.add_from_context(ctx, op, "first")
        dlq.add_from_context(ctx, op, "second")
        assert dlq.size == 1
        assert dlq.get_report()[0].reason == "second"

    def test_dlq_entry_to_dict(self):
        """DLQEntry.to_dict() should produce a serializable dict."""
        entry = DLQEntry(
            context=_make_ctx(5),
            operation_name="AbliterationOp",
            reason="NaN detected",
            timestamp=1000.0,
            attempt_count=3,
            last_error="ValueError: bad value",
        )
        d = entry.to_dict()
        assert d["operation"] == "AbliterationOp"
        assert d["reason"] == "NaN detected"
        assert d["attempts"] == 3
        assert d["last_error"] == "ValueError: bad value"
        assert d["layer"] == "model.layers.5."

    def test_retry_all_no_operation_ref(self):
        """retry_all should handle entries with no operation_ref gracefully."""
        dlq = DeadLetterQueue(max_size=10)
        entry = DLQEntry(
            context=_make_ctx(0),
            operation_name="Test",
            reason="test",
            timestamp=time.time(),
            attempt_count=1,
            operation_ref=None,
        )
        with dlq._lock:
            dlq._entries.append(entry)

        results = dlq.retry_all(lambda ctx, op: True)
        assert results == [False]  # Cannot retry without operation_ref


# ═══════════════════════════════════════════════════════════════
# Overflow strategies (DROP_LOWEST, BLOCK)
# ═══════════════════════════════════════════════════════════════

class TestOverflowStrategies:
    """Test OverflowManager with different strategies."""

    def test_drop_lowest_evicts_least_urgent(self):
        """DROP_LOWEST should evict the item with highest effective_priority."""
        mgr = OverflowManager(
            strategy=OverflowStrategy.DROP_LOWEST,
            max_queue_size=3,
        )
        # Create queue items with different priorities (lower = more urgent)
        items = [
            _make_queue_item(priority=1.0),  # Most urgent
            _make_queue_item(priority=5.0),  # Medium
            _make_queue_item(priority=10.0), # Least urgent
        ]

        new_item = _make_queue_item(priority=2.0)
        dropped = mgr.handle(items, new_item)

        # The least urgent (priority=10.0) should be dropped
        assert dropped is not None
        assert dropped.priority == 10.0
        assert len(items) == 2  # One was popped

    def test_drop_lowest_drops_new_if_least_urgent(self):
        """If new item is less urgent than all queued, drop the new item."""
        mgr = OverflowManager(
            strategy=OverflowStrategy.DROP_LOWEST,
            max_queue_size=3,
        )
        items = [
            _make_queue_item(priority=1.0),
            _make_queue_item(priority=2.0),
            _make_queue_item(priority=3.0),
        ]

        new_item = _make_queue_item(priority=100.0)  # Very low priority
        dropped = mgr.handle(items, new_item)

        assert dropped is new_item
        assert len(items) == 3  # Queue unchanged

    def test_drop_lowest_no_overflow(self):
        """When queue is not full, handle returns None."""
        mgr = OverflowManager(
            strategy=OverflowStrategy.DROP_LOWEST,
            max_queue_size=5,
        )
        items = [_make_queue_item(priority=1.0)]
        new_item = _make_queue_item(priority=2.0)
        result = mgr.handle(items, new_item)
        assert result is None

    def test_block_strategy_falls_back_on_timeout(self):
        """BLOCK should fall back to DROP_LOWEST after timeout."""
        mgr = OverflowManager(
            strategy=OverflowStrategy.BLOCK,
            max_queue_size=2,
            block_timeout=0.1,  # Very short timeout for testing
        )
        items = [
            _make_queue_item(priority=1.0),
            _make_queue_item(priority=5.0),
        ]
        new_item = _make_queue_item(priority=2.0)

        dropped = mgr.handle(items, new_item)
        # After timeout, should fall back to DROP_LOWEST
        assert dropped is not None

    def test_overflow_stats(self):
        mgr = OverflowManager(
            strategy=OverflowStrategy.DROP_LOWEST,
            max_queue_size=2,
        )
        items = [
            _make_queue_item(priority=1.0),
            _make_queue_item(priority=5.0),
        ]
        new_item = _make_queue_item(priority=2.0)
        mgr.handle(items, new_item)

        stats = mgr.get_stats()
        assert stats["dropped_count"] == 1
        assert stats["strategy"] == "drop_lowest"

    def test_spill_disk_strategy(self):
        """SPILL_DISK should write metadata to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = OverflowManager(
                strategy=OverflowStrategy.SPILL_DISK,
                max_queue_size=2,
                spill_dir=tmpdir,
            )
            items = [
                _make_queue_item(priority=1.0),
                _make_queue_item(priority=5.0),
            ]
            new_item = _make_queue_item(priority=2.0)
            mgr.handle(items, new_item)

            assert mgr.spilled_count == 1
            # Check that a spill file was created
            spill_files = list(Path(tmpdir).glob("spill_*.json"))
            assert len(spill_files) == 1

            # Verify content is valid JSON
            with open(spill_files[0]) as f:
                data = json.load(f)
            assert "operation" in data
            assert "priority" in data


# ═══════════════════════════════════════════════════════════════
# CheckpointManager with corrupted JSON
# ═══════════════════════════════════════════════════════════════

class TestCheckpointManagerCorrupted:
    """Test CheckpointManager recovery from corrupted checkpoint files."""

    def test_corrupted_json_starts_fresh(self):
        """Corrupted checkpoint file should result in fresh start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write corrupted JSON
            ckpt_file = Path(tmpdir) / ".aegis_checkpoint.json"
            ckpt_file.write_text("THIS IS NOT JSON {{{{", encoding="utf-8")

            # Should not crash, just starts fresh
            ckpt = CheckpointManager(output_path=tmpdir)
            assert ckpt.num_completed == 0

    def test_empty_file_starts_fresh(self):
        """Empty checkpoint file should result in fresh start."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = Path(tmpdir) / ".aegis_checkpoint.json"
            ckpt_file.write_text("", encoding="utf-8")

            ckpt = CheckpointManager(output_path=tmpdir)
            assert ckpt.num_completed == 0

    def test_missing_keys_starts_fresh(self):
        """Checkpoint with unexpected structure should still work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_file = Path(tmpdir) / ".aegis_checkpoint.json"
            ckpt_file.write_text('{"unexpected": true}', encoding="utf-8")

            ckpt = CheckpointManager(output_path=tmpdir)
            assert ckpt.num_completed == 0

    def test_run_id_mismatch_starts_fresh(self):
        """Different run_id should start fresh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt1 = CheckpointManager(output_path=tmpdir, run_id="run_001")
            ckpt1.mark_completed("model.layers.0.")
            assert ckpt1.num_completed == 1

            ckpt2 = CheckpointManager(output_path=tmpdir, run_id="run_002")
            assert ckpt2.num_completed == 0

    def test_no_run_id_accepts_any(self):
        """When no run_id is specified, accept any checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt1 = CheckpointManager(output_path=tmpdir, run_id="run_001")
            ckpt1.mark_completed("model.layers.0.")

            ckpt2 = CheckpointManager(output_path=tmpdir)  # No run_id
            assert ckpt2.num_completed == 1

    def test_nonexistent_output_dir(self):
        """CheckpointManager should create output dir on first save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = Path(tmpdir) / "does_not_exist"
            ckpt = CheckpointManager(output_path=new_dir)
            ckpt.mark_completed("model.layers.0.")
            assert (new_dir / ".aegis_checkpoint.json").exists()


# ═══════════════════════════════════════════════════════════════
# LayerContext for edge layer indices (0, 47)
# ═══════════════════════════════════════════════════════════════

class TestLayerContextEdgeIndices:
    """Test LayerContext at boundary layer indices."""

    def test_first_layer_index_0(self):
        ctx = _make_ctx(0, "linear_attention")
        assert ctx.layer_index == 0
        assert ctx.layer_fraction == 0.0
        assert ctx.is_deltanet is True
        assert _LAYER_TYPES[0] == "linear_attention"

    def test_last_layer_index_47(self):
        ctx = _make_ctx(47, "full_attention")
        assert ctx.layer_index == 47
        assert ctx.layer_fraction == pytest.approx(47.0 / 47.0)  # == 1.0
        assert ctx.is_rope_enabled is True
        assert _LAYER_TYPES[47] == "full_attention"

    def test_layer_fraction_single_layer_model(self):
        """A model with 1 layer should have fraction 0.0."""
        ctx = LayerContext(
            layer_index=0,
            layer_name="model.layers.0.",
            layer_type="linear_attention",
            total_layers=1,
        )
        assert ctx.layer_fraction == 0.0

    def test_layer_fraction_two_layer_model(self):
        """Index 1 of a 2-layer model should have fraction 1.0."""
        ctx = LayerContext(
            layer_index=1,
            layer_name="model.layers.1.",
            layer_type="full_attention",
            total_layers=2,
        )
        assert ctx.layer_fraction == pytest.approx(1.0)

    def test_repr_includes_key_info(self):
        ctx = _make_ctx(0)
        r = repr(ctx)
        assert "index=0" in r
        assert "linear_attention" in r

    def test_from_config_with_mock_config(self):
        """from_config should work with a mock config object."""
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 48
        mock_config.num_experts = 512
        mock_config.num_activated_experts = 10
        mock_config.hidden_size = 2048
        mock_config.head_dim = 256
        mock_config.num_attention_heads = 16
        mock_config.num_key_value_heads = 2
        mock_config.layer_types = None

        ctx = LayerContext.from_config(
            index=3,
            config=mock_config,
            checkpoint_path="/tmp/test",
        )
        assert ctx.layer_index == 3
        assert ctx.layer_type == "full_attention"  # Index 3 is full_attention
        assert ctx.total_layers == 48
        assert ctx.num_experts == 512


# ═══════════════════════════════════════════════════════════════
# Operation apply() with mock tensors
# ═══════════════════════════════════════════════════════════════

class TestOperationApply:
    """Test apply() methods with actual torch tensors."""

    def test_abliteration_applies_to_matching_keys(self):
        directions = torch.randn(1, 2048)
        op = AbliterationOp(refusal_directions=directions)
        ctx = _make_ctx(0, "linear_attention")

        state_dict = {
            "self_attn.o_proj.weight": torch.randn(2048, 4096, dtype=torch.float16),
            "mlp.experts.0.down_proj.weight": torch.randn(2048, 1024, dtype=torch.float16),
            "mlp.gate.weight": torch.randn(512, 2048, dtype=torch.float16),  # Not targeted
        }
        original_gate = state_dict["mlp.gate.weight"].clone()

        result = op.apply(state_dict, ctx)
        # gate should be unchanged (not targeted)
        assert torch.equal(result["mlp.gate.weight"], original_gate)
        # o_proj and down_proj should be modified
        assert op._tensors_modified >= 2

    def test_weight_inspect_does_not_modify(self):
        op = WeightInspectOp()
        ctx = _make_ctx(0)
        state_dict = {
            "w1": torch.randn(10, 10),
            "w2": torch.randn(5, 5),
        }
        originals = {k: v.clone() for k, v in state_dict.items()}
        result = op.apply(state_dict, ctx)
        for k in originals:
            assert torch.equal(result[k], originals[k])

    def test_quantize_casts_dtype(self):
        op = QuantizeOp(target_dtype=torch.float16)
        ctx = _make_ctx(0)
        state_dict = {
            "w": torch.randn(10, 10, dtype=torch.float32),
        }
        result = op.apply(state_dict, ctx)
        assert result["w"].dtype == torch.float16

    def test_quantize_skips_already_correct_dtype(self):
        op = QuantizeOp(target_dtype=torch.float16)
        ctx = _make_ctx(0)
        state_dict = {
            "w": torch.randn(10, 10, dtype=torch.float16),
        }
        result = op.apply(state_dict, ctx)
        assert result["w"].dtype == torch.float16

    def test_expert_prune_zeros_targeted_experts(self):
        op = ExpertPruneOp(experts_to_prune=[0], prune_mode="zero")
        ctx = _make_ctx(0)
        state_dict = {
            "mlp.experts.0.down_proj.weight": torch.randn(2048, 1024),
            "mlp.experts.1.down_proj.weight": torch.randn(2048, 1024),
        }
        original_expert1 = state_dict["mlp.experts.1.down_proj.weight"].clone()

        result = op.apply(state_dict, ctx)
        # Expert 0 should be zeroed
        assert torch.all(result["mlp.experts.0.down_proj.weight"] == 0)
        # Expert 1 should be unchanged
        assert torch.equal(result["mlp.experts.1.down_proj.weight"], original_expert1)

    def test_expert_prune_scale_mode(self):
        op = ExpertPruneOp(experts_to_prune=[0], prune_mode="scale", scale_factor=0.5)
        ctx = _make_ctx(0)
        original = torch.randn(2048, 1024)
        state_dict = {
            "mlp.experts.0.down_proj.weight": original.clone(),
        }
        result = op.apply(state_dict, ctx)
        expected = original * 0.5
        assert torch.allclose(result["mlp.experts.0.down_proj.weight"], expected)

    def test_longrope_with_inv_freq(self):
        factors = torch.ones(128)
        op = LongRoPEOp(rescale_factors=factors)
        ctx = _make_ctx(3, "full_attention")

        state_dict = {
            "self_attn.rotary_emb.inv_freq": torch.randn(128),
        }
        result = op.apply(state_dict, ctx)
        assert "self_attn.rotary_emb.inv_freq" in result
        assert result["self_attn.rotary_emb.inv_freq"].shape == (128,)

    def test_lora_merge_applies_delta(self):
        # Create LoRA adapter with weights for layer 0
        lora_dict = {
            "layers.0.self_attn.q_proj.lora_A.weight": torch.randn(16, 2048),
            "layers.0.self_attn.q_proj.lora_B.weight": torch.randn(2048, 16),
        }
        op = LoRAMergeOp(lora_state_dict=lora_dict, lora_alpha=32, lora_rank=16)
        ctx = _make_ctx(0)

        state_dict = {
            "self_attn.q_proj.weight": torch.randn(2048, 2048, dtype=torch.float16),
        }
        original = state_dict["self_attn.q_proj.weight"].clone()

        result = op.apply(state_dict, ctx)
        # Should be modified (delta added)
        assert not torch.equal(result["self_attn.q_proj.weight"], original)


# ═══════════════════════════════════════════════════════════════
# QQMSQueue execute() retry and DLQ routing
# ═══════════════════════════════════════════════════════════════

class TestQQMSQueueExecution:
    """Test QQMS queue execute with retries and DLQ."""

    def test_successful_execution(self):
        queue = QQMSQueue(config=QQMSConfig())
        mock_op = MagicMock()
        mock_op.apply = MagicMock(return_value={"w": torch.randn(4, 4)})
        type(mock_op).__name__ = "MockOp"

        item = QueueItem(
            operation=mock_op,
            state_dict={"w": torch.randn(4, 4)},
            context=_make_ctx(0),
            priority=1.0,
            max_attempts=3,
        )

        result = queue.execute(item)
        assert "w" in result
        assert queue.get_stats().items_processed == 1

    def test_exhausted_retries_goes_to_dlq(self):
        queue = QQMSQueue(config=QQMSConfig())
        mock_op = MagicMock()
        mock_op.apply = MagicMock(side_effect=RuntimeError("fail"))
        type(mock_op).__name__ = "FailOp"

        item = QueueItem(
            operation=mock_op,
            state_dict={"w": torch.randn(4, 4)},
            context=_make_ctx(0),
            priority=1.0,
            max_attempts=1,  # Will exhaust on first failure
        )

        with pytest.raises(RuntimeError, match="failed after"):
            queue.execute(item)

        assert queue.dlq.size == 1
        assert queue.get_stats().items_dlq == 1

    def test_retry_then_succeed(self):
        """Operation fails once, then succeeds on retry."""
        queue = QQMSQueue(config=QQMSConfig())
        mock_op = MagicMock()
        call_count = 0

        def side_effect_fn(state_dict, ctx):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return {"w": torch.randn(4, 4)}

        mock_op.apply = side_effect_fn
        type(mock_op).__name__ = "RetryOp"

        item = QueueItem(
            operation=mock_op,
            state_dict={"w": torch.randn(4, 4)},
            context=_make_ctx(0),
            priority=1.0,
            max_attempts=3,
        )

        # Patch time.sleep to avoid actual waiting during retry backoff
        with patch("aegis_trainer.queue.qqms.time.sleep"):
            result = queue.execute(item)
        assert "w" in result
        stats = queue.get_stats()
        assert stats.items_processed == 1
        assert stats.items_failed == 1
        assert stats.items_retried == 1

    def test_send_to_dlq_direct(self):
        queue = QQMSQueue(config=QQMSConfig())
        ctx = _make_ctx(0)
        op = WeightInspectOp()

        queue.send_to_dlq(ctx, op, "Manual DLQ send")
        assert queue.dlq.size == 1
        assert queue.get_stats().items_dlq == 1

    def test_enqueue_and_execute_pending(self):
        queue = QQMSQueue(config=QQMSConfig())

        mock_op = MagicMock()
        mock_op.apply = MagicMock(return_value={"w": torch.randn(4, 4)})
        type(mock_op).__name__ = "MockOp"

        for i in range(3):
            item = QueueItem(
                operation=mock_op,
                state_dict={"w": torch.randn(4, 4)},
                context=_make_ctx(i),
                priority=float(i),
                max_attempts=3,
            )
            queue.enqueue(item)

        assert queue.pending_count == 3
        results = queue.execute_pending()
        assert len(results) == 3
        assert queue.pending_count == 0


# ═══════════════════════════════════════════════════════════════
# Profiler edge cases
# ═══════════════════════════════════════════════════════════════

class TestProfilerEdgeCases:
    """Test profiler with unusual usage patterns."""

    def test_auto_end_previous_layer(self):
        """Starting a new layer while previous is active auto-ends it."""
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        profiler.start_layer("model.layers.1.")  # Auto-ends layer 0
        assert profiler.completed_layers == 1
        profiler.end_layer("model.layers.1.")
        assert profiler.completed_layers == 2

    def test_auto_end_operation_on_layer_end(self):
        """Operations still running when layer ends are auto-ended."""
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        profiler.start_operation("load")
        # Not calling end_operation("load")
        profiler.end_layer("model.layers.0.")  # Should auto-end "load"
        assert profiler.completed_layers == 1

    def test_layers_per_minute_initially_zero(self):
        profiler = TrainerProfiler()
        assert profiler.layers_per_minute == 0.0

    def test_avg_layer_time_initially_zero(self):
        profiler = TrainerProfiler()
        assert profiler.avg_layer_time == 0.0

    def test_estimated_remaining_no_data(self):
        profiler = TrainerProfiler()
        assert profiler.estimated_remaining(48) == 0.0

    def test_estimated_remaining_all_done(self):
        profiler = TrainerProfiler()
        profiler.start_layer("model.layers.0.")
        time.sleep(0.01)
        profiler.end_layer("model.layers.0.")
        # If total_layers = 1, remaining = 0
        assert profiler.estimated_remaining(1) == 0.0


# ═══════════════════════════════════════════════════════════════
# Validation edge cases
# ═══════════════════════════════════════════════════════════════

class TestValidationEdgeCases:
    """Test operation validation with edge-case tensors."""

    def test_base_validation_detects_nan(self):
        from aegis_trainer.ops.base import LayerOperation

        # Create a concrete subclass
        class TestOp(LayerOperation):
            name = "test"
            def should_apply(self, ctx):
                return True
            def apply(self, state_dict, ctx):
                return state_dict

        op = TestOp()
        ctx = _make_ctx(0)
        modified = {"w": torch.tensor([float("nan"), 1.0])}
        assert op.validate({}, modified, ctx) is False

    def test_base_validation_detects_inf(self):
        from aegis_trainer.ops.base import LayerOperation

        class TestOp(LayerOperation):
            name = "test"
            def should_apply(self, ctx):
                return True
            def apply(self, state_dict, ctx):
                return state_dict

        op = TestOp()
        ctx = _make_ctx(0)
        modified = {"w": torch.tensor([float("inf"), 1.0])}
        assert op.validate({}, modified, ctx) is False

    def test_base_validation_passes_clean_tensors(self):
        from aegis_trainer.ops.base import LayerOperation

        class TestOp(LayerOperation):
            name = "test"
            def should_apply(self, ctx):
                return True
            def apply(self, state_dict, ctx):
                return state_dict

        op = TestOp()
        ctx = _make_ctx(0)
        modified = {"w": torch.randn(10, 10)}
        assert op.validate({}, modified, ctx) is True

    def test_quantize_validation_dtype_check(self):
        op = QuantizeOp(target_dtype=torch.float16)
        ctx = _make_ctx(0)
        # Modified dict has wrong dtype
        original = {"w": torch.randn(10, 10, dtype=torch.float32)}
        modified = {"w": torch.randn(10, 10, dtype=torch.float32)}  # Still f32!
        assert op.validate(original, modified, ctx) is False

    def test_quantize_validation_passes_correct_dtype(self):
        op = QuantizeOp(target_dtype=torch.float16)
        ctx = _make_ctx(0)
        original = {"w": torch.randn(10, 10, dtype=torch.float32)}
        modified = {"w": torch.randn(10, 10, dtype=torch.float16)}
        assert op.validate(original, modified, ctx) is True
