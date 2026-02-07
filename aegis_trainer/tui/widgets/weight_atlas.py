"""
WeightAtlas — Braille scatter plot of neural network weight distributions.

Renders model weights as colored dots in 2D space using PCA projection,
creating an Apple Embedding Atlas-like visualization in the terminal.
Each dot represents sampled weight values, color-coded by tensor type:
  - Attention: cyan (#00d4ff)
  - MoE Experts: yellow (#ffd700)
  - Shared Expert: orange (#ff8800)
  - Layer Norms: green (#00ff88)
  - Router/Gate: purple (#aa88ff)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import math
import random
from typing import Any

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from aegis_trainer.tui.widgets.braille_canvas import BrailleCanvas

# ---------------------------------------------------------------------------
# Optional torch import — gracefully degrade when unavailable
# ---------------------------------------------------------------------------
try:
    import torch
    from torch import Tensor

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    Tensor = Any  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Tensor-type color mapping
# ---------------------------------------------------------------------------

TENSOR_COLORS: dict[str, str] = {
    "attention": "#00d4ff",
    "experts": "#ffd700",
    "shared": "#ff8800",
    "norms": "#00ff88",
    "gate": "#aa88ff",
    "other": "#6a7a8a",
}

TENSOR_COLORS_DIM: dict[str, str] = {
    "attention": "#006680",
    "experts": "#806c00",
    "shared": "#804400",
    "norms": "#006644",
    "gate": "#554488",
    "other": "#353d44",
}

# Human-readable labels for the legend
_LABELS: dict[str, str] = {
    "attention": "attn",
    "experts": "experts",
    "shared": "shared",
    "norms": "norms",
    "gate": "gate",
    "other": "other",
}

# Maximum total weight samples to collect across all tensors
_MAX_SAMPLES = 50_000

# Default canvas dimensions (chars)
_CANVAS_W = 60
_CANVAS_H = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_tensor(key: str) -> str:
    """Classify a state-dict key into a tensor category.

    Args:
        key: Fully-qualified parameter name (e.g. ``"layers.0.self_attn.q_proj.weight"``).

    Returns:
        Category string matching a key in :data:`TENSOR_COLORS`.
    """
    k = key.lower()
    if "self_attn" in k:
        return "attention"
    if "mlp.experts" in k:
        return "experts"
    if "mlp.shared_expert" in k:
        return "shared"
    if "layernorm" in k:
        return "norms"
    if "mlp.gate" in k:
        return "gate"
    return "other"


def _sample_weights(
    state_dict: dict[str, Any],
) -> tuple[Any, list[str]]:
    """Sample up to ``_MAX_SAMPLES`` weight values from a state dict.

    Sampling is proportional: larger tensors contribute more samples than
    smaller ones.

    Args:
        state_dict: Mapping of parameter names to tensors (or array-likes).

    Returns:
        A tuple of ``(samples_1d, types)`` where *samples_1d* is a 1-D
        tensor of sampled values and *types* is a parallel list of tensor
        category strings.
    """
    if not _HAS_TORCH:
        return _sample_weights_fallback(state_dict)

    # Compute total elements across all tensors
    total_elements = 0
    tensor_info: list[tuple[str, str, torch.Tensor]] = []
    for key, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue
        flat = param.detach().float().flatten()
        if flat.numel() == 0:
            continue
        category = _classify_tensor(key)
        tensor_info.append((key, category, flat))
        total_elements += flat.numel()

    if total_elements == 0:
        return torch.zeros(0), []

    all_samples: list[torch.Tensor] = []
    all_types: list[str] = []

    for _key, category, flat in tensor_info:
        # Proportional sample count
        n = max(1, int(_MAX_SAMPLES * flat.numel() / total_elements))
        n = min(n, flat.numel())

        indices = torch.randint(0, flat.numel(), (n,))
        sampled = flat[indices]
        all_samples.append(sampled)
        all_types.extend([category] * n)

    samples_1d = torch.cat(all_samples) if all_samples else torch.zeros(0)
    return samples_1d, all_types


def _sample_weights_fallback(
    state_dict: dict[str, Any],
) -> tuple[list[float], list[str]]:
    """Pure-Python fallback when torch is unavailable."""
    total_elements = 0
    tensor_info: list[tuple[str, list[float]]] = []
    for key, param in state_dict.items():
        try:
            vals = list(param.flatten()) if hasattr(param, "flatten") else list(param)
            vals = [float(v) for v in vals]
        except (TypeError, ValueError):
            continue
        if not vals:
            continue
        category = _classify_tensor(key)
        tensor_info.append((category, vals))
        total_elements += len(vals)

    if total_elements == 0:
        return [], []

    all_samples: list[float] = []
    all_types: list[str] = []

    for category, vals in tensor_info:
        n = max(1, int(_MAX_SAMPLES * len(vals) / total_elements))
        n = min(n, len(vals))
        indices = random.sample(range(len(vals)), n)
        for idx in indices:
            all_samples.append(vals[idx])
            all_types.append(category)

    return all_samples, all_types


def _project_2d(
    samples: Any,
) -> Any:
    """Project 1-D weight samples into 2-D coordinates via SVD.

    Creates paired features ``(value, |value|)`` and applies a rank-2 SVD
    to get 2-D coordinates suitable for scatter plotting.

    Args:
        samples: 1-D tensor of weight values.

    Returns:
        Tensor of shape ``[N, 2]`` with projected coordinates.
    """
    if not _HAS_TORCH:
        return _project_2d_fallback(samples)

    n = samples.numel()
    if n == 0:
        return torch.zeros(0, 2)

    # Build [N, 2] feature matrix: (value, abs_value)
    features = torch.stack([samples, samples.abs()], dim=1).float()

    # Center the data
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean

    try:
        U, S, V = torch.svd_lowrank(centered, q=2)
        coords = U[:, :2] * S[:2].unsqueeze(0)
    except Exception:
        # Fallback: use raw (value, abs_value) — still gives useful spread
        coords = centered

    return coords


def _project_2d_fallback(
    samples: list[float],
) -> list[tuple[float, float]]:
    """Pure-Python 2D projection fallback."""
    if not samples:
        return []

    # Simple approach: (value, abs(value)) centered
    mean_v = sum(samples) / len(samples)
    abs_vals = [abs(s) for s in samples]
    mean_a = sum(abs_vals) / len(abs_vals)

    return [(s - mean_v, abs(s) - mean_a) for s in samples]


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class WeightAtlas(Widget):
    """Textual widget rendering an Apple Embedding Atlas-style scatter plot
    of neural network weight distributions using braille characters.

    The plot is updated by calling :meth:`update_weights` with a model
    state dict.  Weights are sampled, projected to 2-D via SVD, and
    rendered as colored braille dots.
    """

    DEFAULT_CSS = """
    WeightAtlas {
        height: 1fr;
        background: #0a0e14;
        border: solid #2a3a4a;
        padding: 1;
    }
    WeightAtlas .wa-title {
        color: #00d4ff;
        text-style: bold;
        padding: 0 0 1 0;
    }
    WeightAtlas #wa-canvas {
        padding: 0;
    }
    WeightAtlas #wa-legend {
        color: #6a7a8a;
        padding: 1 0 0 0;
    }
    WeightAtlas #wa-stats {
        color: #6a7a8a;
        padding: 0;
    }
    """

    layer_index: reactive[int] = reactive(-1)
    is_modified: reactive[bool] = reactive(False)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        # Store previous projection data for before/after comparison
        self._prev_coords: Any = None
        self._prev_types: list[str] = []

    def compose(self) -> ComposeResult:
        yield Static("Weight Atlas", classes="wa-title")
        yield Static("", id="wa-canvas")
        yield Static(self._render_legend(), id="wa-legend")
        yield Static("", id="wa-stats")

    # ------------------------------------------------------------------
    # Public update methods
    # ------------------------------------------------------------------

    def update_weights(
        self,
        state_dict: dict[str, Any],
        ctx: Any = None,
    ) -> None:
        """Sample, project, and render weight distribution from *state_dict*.

        Args:
            state_dict: Model parameter dict (keys -> tensors).
            ctx: Optional context object (unused, reserved for future use).
        """
        samples, types = _sample_weights(state_dict)
        coords = _project_2d(samples)

        canvas = BrailleCanvas(_CANVAS_W, _CANVAS_H)
        self._plot_on_canvas(canvas, coords, types, TENSOR_COLORS)

        rendered = canvas.render()
        self._update_canvas(rendered)
        self._update_stats(types, dim=False)

        # Store for before/after overlay
        self._prev_coords = coords
        self._prev_types = list(types)
        self.is_modified = False

    def update_modified(
        self,
        state_dict: dict[str, Any],
        ctx: Any = None,
    ) -> None:
        """Overlay modified weights on top of original (dim) weights.

        Call :meth:`update_weights` first to set the baseline, then call
        this method with the modified state dict.  Original weights appear
        in dim colors; modified weights appear in bright colors.

        Args:
            state_dict: Modified model parameter dict.
            ctx: Optional context object (unused).
        """
        canvas = BrailleCanvas(_CANVAS_W, _CANVAS_H)

        # Plot original weights in dim colors (if available)
        if self._prev_coords is not None and self._prev_types:
            self._plot_on_canvas(
                canvas, self._prev_coords, self._prev_types, TENSOR_COLORS_DIM,
            )

        # Project and plot modified weights in bright colors
        samples, types = _sample_weights(state_dict)
        coords = _project_2d(samples)
        self._plot_on_canvas(canvas, coords, types, TENSOR_COLORS)

        rendered = canvas.render()
        self._update_canvas(rendered)
        self._update_stats(types, dim=True)
        self.is_modified = True

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _plot_on_canvas(
        self,
        canvas: BrailleCanvas,
        coords: Any,
        types: list[str],
        color_map: dict[str, str],
    ) -> None:
        """Plot projected coordinates onto a :class:`BrailleCanvas`.

        Args:
            canvas: Target canvas.
            coords: 2-D coordinates (torch Tensor or list of tuples).
            types: Parallel list of tensor category strings.
            color_map: Category -> hex color mapping.
        """
        if _HAS_TORCH and isinstance(coords, torch.Tensor):
            if coords.numel() == 0:
                return
            xs = coords[:, 0]
            ys = coords[:, 1]
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
            for i in range(coords.size(0)):
                color = color_map.get(types[i], color_map.get("other", "#6a7a8a"))
                canvas.set_point(
                    float(xs[i]),
                    float(ys[i]),
                    color=color,
                    x_range=(x_min, x_max),
                    y_range=(y_min, y_max),
                )
        else:
            # Fallback for list-of-tuples
            if not coords:
                return
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            for i, (cx, cy) in enumerate(coords):
                color = color_map.get(types[i], color_map.get("other", "#6a7a8a"))
                canvas.set_point(
                    cx,
                    cy,
                    color=color,
                    x_range=(x_min, x_max),
                    y_range=(y_min, y_max),
                )

    def _update_canvas(self, rendered: str) -> None:
        """Push rendered braille text into the canvas Static widget."""
        try:
            widget = self.query_one("#wa-canvas", Static)
            widget.update(rendered)
        except Exception:
            pass

    def _update_stats(self, types: list[str], dim: bool = False) -> None:
        """Update the stats line below the plot."""
        try:
            widget = self.query_one("#wa-stats", Static)
        except Exception:
            return

        total = len(types)
        if total == 0:
            widget.update("[#6a7a8a]No weights loaded[/]")
            return

        # Count per category
        counts: dict[str, int] = {}
        for t in types:
            counts[t] = counts.get(t, 0) + 1

        parts: list[str] = [f"[#6a7a8a]{total:,} samples[/]"]
        if dim:
            parts.append("[#6a7a8a]  (before/after overlay)[/]")

        # Top categories
        for cat in ("attention", "experts", "shared", "norms", "gate"):
            n = counts.get(cat, 0)
            if n > 0:
                color = TENSOR_COLORS[cat]
                pct = 100.0 * n / total
                parts.append(f"  [{color}]{_LABELS[cat]}: {pct:.0f}%[/]")

        widget.update(" ".join(parts))

    @staticmethod
    def _render_legend() -> str:
        """Build horizontal color legend with Rich markup."""
        items: list[str] = []
        for cat in ("attention", "experts", "shared", "norms", "gate"):
            color = TENSOR_COLORS[cat]
            label = _LABELS[cat]
            items.append(f"[{color}]\u25cf[/] {label}")
        return "  ".join(items)
