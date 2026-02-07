<p align="center">
  <img src="https://img.shields.io/badge/license-SSPL--1.0-blue?style=for-the-badge" alt="License">
  <img src="https://img.shields.io/badge/python-3.10+-green?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/GPU-Intel%20Arc%20%7C%20Vulkan-00d4ff?style=for-the-badge" alt="GPU">
  <img src="https://img.shields.io/badge/MoE-80B%20params-ffd700?style=for-the-badge" alt="MoE">
</p>

<h1 align="center">AEGIS AI Trainer</h1>

<p align="center">
  Layer-by-layer model training and modification framework for 80B+ parameter MoE models on consumer GPUs.
  <br/>
  Built because nothing else could do it.
</p>

<p align="center">
  <a href="https://justcalljon.pro">justcalljon.pro</a> &middot;
  <a href="#features">Features</a> &middot;
  <a href="#architecture">Architecture</a> &middot;
  <a href="#installation">Install</a> &middot;
  <a href="#usage">Usage</a>
</p>

---

## The Problem

You want to modify an 80B parameter MoE model. Maybe you need to abliterate refusal vectors, extend context to 512K, or merge a LoRA adapter into the base weights. Cool.

Except:

- **AirLLM** streams layers for inference, but can't modify weights or write them back. It's read-only by design. ([see issue](https://github.com/lyogavin/airllm/issues/))
- **Heretic** does abliteration, but loads the entire model into memory. 80B at BF16 = 160GB minimum. Good luck fitting that on a consumer GPU. ([see issue](https://github.com/p-e-w/heretic/issues/))
- **Unsloth/PEFT** train LoRA adapters but don't merge them layer-by-layer. You're stuck if you can't fit the full model in VRAM.

So we built one that actually works.

## The Solution

AEGIS Trainer streams model layers from NVMe through RAM to VRAM one at a time, applies operations to each layer, and writes the modified weights back. Your 80B MoE model never fully loads into memory. 11GB VRAM is enough.

```
NVMe (safetensors) --> 120GB RAM --> 11GB VRAM --> modify --> write back
                       one layer at a time
```

## Features

### Operations
- **Abliteration** -- Directional ablation to remove refusal vectors. Targets all 512 experts per layer, not just the dense projection. Adapted from Heretic's approach but runs layer-by-layer instead of loading 160GB into memory.
- **LongRoPE** -- Context window extension from 256K to 512K tokens. Only touches the 12 full-attention layers (DeltaNet layers don't use RoPE).
- **LoRA Merge** -- Merge PEFT/LoRA adapters directly into base weights. Per-layer `W_merged = W_original + (B @ A) * scaling`.
- **Expert Pruning** -- Zero out or scale specific MoE experts. Useful for removing underperforming routed experts.
- **Weight Inspection** -- Non-destructive per-tensor statistics: min, max, mean, std, sparsity. Generates layer-by-layer reports.
- **Quantization** -- Per-layer dtype casting for compression.

### Architecture
- **QQMS Queue** -- Priority queue with age-boosted scheduling, dead letter queue, and overflow management. Resource-aware throttling prevents OOM.
- **Checkpoint Manager** -- Atomic JSON persistence. Resume interrupted runs from the exact layer that failed.
- **Resource Monitor** -- Real-time RAM, VRAM, and CPU tracking. Intel Arc B580 safe (uses `torch.xpu` and sysfs, never `torch.cuda`).
- **Layer IO** -- Composition-based safetensors load/save following AirLLM naming conventions.

### TUI (Terminal UI)
Full Textual-based terminal interface with 7 tabs:

| Tab | Key | Purpose |
|-----|-----|---------|
| Dashboard | `1` | Resource monitoring, progress, queue summary |
| Models | `2` | Browse and inspect available models (HF + GGUF) |
| Build | `3` | Step-by-step operation configuration wizard |
| Inspect | `4` | Per-layer weight inspector with arrow-key navigation |
| Queue | `5` | Queue management -- reorder, cancel, pause/resume |
| Logs | `6` | Scrollable, filterable log viewer |
| Visualizer | `7` | Live weight visualization during operations |

### Weight Visualization
The Visualizer tab renders live weight distributions using Unicode braille scatter plots (2x4 dot resolution per character cell) with per-tensor-type color coding:

- **Cyan** -- Attention projections (q/k/v/o)
- **Yellow** -- MoE expert weights
- **Orange** -- Shared expert weights
- **Green** -- Layer norms
- **Purple** -- Gating network

Includes before/after overlay during modifications and a Heretic-style step-by-step progress display.

### CLI
```bash
aegis-trainer run \
  --model /path/to/model \
  --output /path/to/output \
  --op abliterate \
  --op longrope:target_length=524288

aegis-trainer inspect --model /path/to/model
aegis-trainer queue --status
aegis-trainer tui
```

## Architecture

```
aegis_trainer/
  __init__.py             # Public API exports
  layer_trainer.py        # Core orchestrator (878 lines)
  layer_context.py        # Frozen dataclass per-layer metadata
  cli.py                  # Click CLI (673 lines)
  ops/
    base.py               # LayerOperation ABC
    abliteration.py       # Directional ablation (all 512 experts)
    longrope.py           # RoPE extension (12/48 layers)
    lora_merge.py         # LoRA adapter merge
    expert_prune.py       # MoE expert pruning
    weight_inspect.py     # Non-destructive stats
    quantize.py           # Dtype casting
  queue/
    qqms.py               # Resource-aware execution engine
    queue_item.py         # Age-boosted priority scheduling
    dlq.py                # Dead letter queue
    overflow.py           # DROP_LOWEST / BLOCK / SPILL_DISK
  utils/
    resource_monitor.py   # RAM/VRAM/CPU (Intel Arc safe)
    layer_io.py           # Safetensors load/save
    checkpoint.py         # Atomic JSON persistence
    profiler.py           # Layer/operation timing + ETA
  tui/
    app.py                # 7-tab Textual application
    theme.py              # AEGIS color scheme
    screens/              # Dashboard, Models, Build, Inspect, Queue, Logs, Visualizer
    widgets/              # ResourceBar, ProgressPanel, LayerMap, BrailleCanvas, WeightAtlas, WeightHistogram, HereticProgress
```

## Qwen3-Next Support

First-class support for the Qwen3-Next hybrid DeltaNet + Attention architecture:

- **48 layers**: 36 DeltaNet (linear attention) + 12 full attention
- **Layer pattern**: 3x DeltaNet, 1x Attention, repeated 12 times
- **512 MoE experts**, 10 active per token
- **Per-layer type awareness**: operations like LongRoPE automatically skip DeltaNet layers (they don't use RoPE)
- **Color-coded UI**: DeltaNet layers in yellow, Attention layers in cyan throughout the TUI

## Installation

```bash
git clone https://github.com/jonhardwick-spec/aegis-trainer.git
cd aegis-trainer
pip install -e .
```

### Requirements
- Python 3.10+
- PyTorch (with Intel XPU support for Arc GPUs, or standard CUDA)
- textual >= 7.0 (for TUI)
- safetensors, transformers, click, rich

## Usage

### Launch the TUI
```bash
aegis-trainer tui
```

### Run abliteration on an 80B model with 11GB VRAM
```bash
aegis-trainer run \
  --model /path/to/Qwen3-Next-80B \
  --output /path/to/output \
  --op abliterate:strength=1.0
```

### Inspect layer weights
```bash
aegis-trainer inspect --model /path/to/model --format table
```

### Programmatic usage
```python
from aegis_trainer import LayerTrainer, ResourceLimits
from aegis_trainer.ops import AbliterationOp, LongRoPEOp

trainer = LayerTrainer(
    model_path="/path/to/model",
    operations=[
        AbliterationOp(strength=1.0),
        LongRoPEOp(target_length=524288),
    ],
    output_path="/path/to/output",
    resource_limits=ResourceLimits(
        max_ram_gb=120,
        max_vram_gb=11,
        max_cpu_cores=8,
    ),
)
result = trainer.run()
print(result.summary())
```

## Hardware Tested

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen Threadripper PRO 5945WX (12-core) |
| RAM | 128GB DDR4 (120GB allocated) |
| GPU | Intel Arc B580 12GB (Battlemage) |
| Storage | 2x 477GB NVMe + 466GB SSD |
| OS | Ubuntu 25.10, Kernel 6.17 |

The entire 80B MoE model streams through this setup at ~2 layers/min during abliteration. MoE experts sit in RAM, attention kernels run on VRAM.

## Contributing

Pull requests are welcome and encouraged. This project exists because existing tools couldn't handle 80B MoE modifications on consumer hardware, and there's a lot more ground to cover.

**Things that need work:**
- CUDA backend testing (built on Intel Arc, needs validation on NVIDIA)
- AMD ROCm support in the resource monitor
- Additional operations (knowledge distillation, GPTQ, AWQ layer-level quant)
- More model architectures (Llama, Mistral, DeepSeek-V3)
- Safetensors index parsing for sharded models
- Performance profiling and optimization on different hardware configs

**How to contribute:**
1. Fork the repo
2. Create a feature branch (`git checkout -b feature/your-thing`)
3. Make your changes, add tests if applicable
4. Open a PR with a clear description of what you changed and why

If you're working on something big, open an issue first so we don't duplicate effort.

No CLA, no bureaucracy. If your code works and makes the project better, it gets merged.

## License

**SSPL-1.0** (Server Side Public License)

If you offer this as a service, you open-source your entire stack. Use it however you want otherwise.

## Author

**Hardwick Software Services**
[justcalljon.pro](https://justcalljon.pro) &middot; [GitHub](https://github.com/jonhardwick-spec)
