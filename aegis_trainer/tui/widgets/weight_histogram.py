"""
WeightHistogram -- Unicode block histogram of weight value distributions.

Renders horizontal bar chart showing the distribution of weight values
using Unicode fractional block characters for smooth sub-character
resolution. Color gradient from center (green) to extremes (red)
highlights outlier values.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import math
from typing import Sequence

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from aegis_trainer.tui.theme import COLORS

# Torch is optional -- allow pure-list usage when not available.
try:
    import torch  # type: ignore[import-untyped]

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False


# 9 levels: empty through full block
_BLOCK_CHARS = " \u258f\u258e\u258d\u258c\u258b\u258a\u2589\u2588"

# Theme colors by distance from mean (in standard deviations)
_COLOR_WITHIN_1 = COLORS["accent_green"]   # #00ff88
_COLOR_WITHIN_2 = COLORS["accent_yellow"]  # #ffd700
_COLOR_WITHIN_3 = "#ff8800"                # orange
_COLOR_BEYOND_3 = COLORS["accent_red"]     # #ff4444


def _color_for_sigma(sigma_distance: float) -> str:
    """Return a theme color based on how many std-devs from the mean."""
    if sigma_distance <= 1.0:
        return _COLOR_WITHIN_1
    if sigma_distance <= 2.0:
        return _COLOR_WITHIN_2
    if sigma_distance <= 3.0:
        return _COLOR_WITHIN_3
    return _COLOR_BEYOND_3


class WeightHistogram(Widget):
    """Unicode block-character histogram of weight value distributions.

    Call :meth:`update_distribution` with a list of floats (or a PyTorch
    tensor) to render.  Optionally use :meth:`set_comparison_mode` to
    overlay before/after distributions.
    """

    DEFAULT_CSS = """
    WeightHistogram {
        height: auto;
        min-height: 12;
        background: #131924;
        border: solid #2a3a4a;
        padding: 1;
    }
    WeightHistogram .wh-title {
        color: #00d4ff;
        text-style: bold;
        padding: 0 0 1 0;
    }
    WeightHistogram #wh-bars {
        color: #d4dae4;
    }
    WeightHistogram #wh-stats {
        color: #6a7a8a;
        padding: 1 0 0 0;
    }
    """

    def __init__(
        self,
        num_bins: int = 40,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._num_bins = max(4, num_bins)
        self._label: str = "Distribution"
        # Histogram data
        self._counts: list[int] = []
        self._edges: list[float] = []
        self._mean: float = 0.0
        self._std: float = 0.0
        self._min_val: float = 0.0
        self._max_val: float = 0.0
        self._sparsity: float = 0.0
        self._total: int = 0
        # Comparison mode
        self._comparison: bool = False
        self._before_counts: list[int] = []
        self._after_counts: list[int] = []

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(self._label, classes="wh-title")
        yield Static("", id="wh-bars")
        yield Static("", id="wh-stats")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_distribution(
        self,
        values: Sequence[float] | list[float],
        label: str = "",
    ) -> None:
        """Compute histogram from *values* and refresh the display.

        Args:
            values: Weight values as a flat list/sequence of floats, or a
                PyTorch ``Tensor`` (will be flattened and converted).
            label: Optional title override.
        """
        # Convert torch tensor to list if needed
        flat = self._to_flat_list(values)
        if not flat:
            return

        self._comparison = False
        if label:
            self._label = label

        self._compute_stats(flat)
        self._compute_bins(flat)
        self._refresh_display()

    def set_comparison_mode(
        self,
        before: Sequence[float] | list[float],
        after: Sequence[float] | list[float],
    ) -> None:
        """Overlay two distributions (before/after) for comparison.

        Before values are rendered with dim colors; after values use bright
        colors.
        """
        before_flat = self._to_flat_list(before)
        after_flat = self._to_flat_list(after)
        if not before_flat or not after_flat:
            return

        self._comparison = True
        all_vals = before_flat + after_flat
        self._compute_stats(all_vals)

        v_min = self._min_val
        v_max = self._max_val
        span = v_max - v_min if v_max != v_min else 1.0

        self._before_counts = [0] * self._num_bins
        for v in before_flat:
            idx = min(int((v - v_min) / span * self._num_bins), self._num_bins - 1)
            self._before_counts[idx] += 1

        self._after_counts = [0] * self._num_bins
        for v in after_flat:
            idx = min(int((v - v_min) / span * self._num_bins), self._num_bins - 1)
            self._after_counts[idx] += 1

        self._edges = [v_min + i * span / self._num_bins for i in range(self._num_bins + 1)]
        self._refresh_display()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_flat_list(values: object) -> list[float]:
        """Convert a list, sequence, or torch.Tensor to a flat list of floats."""
        if _HAS_TORCH and isinstance(values, torch.Tensor):
            return values.detach().cpu().flatten().tolist()
        return list(values)  # type: ignore[arg-type]

    def _compute_stats(self, flat: list[float]) -> None:
        n = len(flat)
        self._total = n
        self._min_val = min(flat)
        self._max_val = max(flat)
        self._mean = sum(flat) / n
        variance = sum((v - self._mean) ** 2 for v in flat) / max(n - 1, 1)
        self._std = math.sqrt(variance)
        self._sparsity = sum(1 for v in flat if abs(v) < 1e-6) / n * 100.0

    def _compute_bins(self, flat: list[float]) -> None:
        v_min, v_max = self._min_val, self._max_val
        span = v_max - v_min if v_max != v_min else 1.0
        self._counts = [0] * self._num_bins
        for v in flat:
            idx = min(int((v - v_min) / span * self._num_bins), self._num_bins - 1)
            self._counts[idx] += 1
        self._edges = [v_min + i * span / self._num_bins for i in range(self._num_bins + 1)]

    def _refresh_display(self) -> None:
        try:
            title_w = self.query_one(".wh-title", Static)
            title_w.update(self._label)
        except Exception:
            pass
        try:
            self.query_one("#wh-bars", Static).update(self._render_histogram())
        except Exception:
            pass
        try:
            self.query_one("#wh-stats", Static).update(self._render_stats())
        except Exception:
            pass

    def _render_histogram(self) -> str:
        """Build Rich-markup string for the horizontal bar chart."""
        if self._comparison:
            return self._render_comparison()

        if not self._counts:
            return "[dim]No data[/dim]"

        max_count = max(self._counts) or 1
        # Allocate bar width: leave room for labels
        bar_width = 30
        lines: list[str] = []

        for i, count in enumerate(self._counts):
            edge_lo = self._edges[i]
            edge_hi = self._edges[i + 1]
            bin_center = (edge_lo + edge_hi) / 2.0

            # Sigma distance from mean
            sigma = abs(bin_center - self._mean) / self._std if self._std > 0 else 0.0
            color = _color_for_sigma(sigma)

            # Bar length in fractional characters
            frac = count / max_count
            full_chars = int(frac * bar_width)
            remainder = (frac * bar_width) - full_chars
            part_idx = int(remainder * 8)

            bar = _BLOCK_CHARS[8] * full_chars
            if part_idx > 0 and full_chars < bar_width:
                bar += _BLOCK_CHARS[part_idx]

            label = f"{edge_lo:+.3f}"
            pct = count / self._total * 100.0 if self._total else 0.0
            lines.append(
                f"[{COLORS['text_muted']}]{label:>8}[/] "
                f"[{color}]{bar}[/]"
                f" [{COLORS['text_muted']}]{pct:5.1f}%[/]"
            )

        return "\n".join(lines)

    def _render_comparison(self) -> str:
        """Render overlapping before/after histograms."""
        if not self._before_counts or not self._after_counts:
            return "[dim]No comparison data[/dim]"

        max_count = max(max(self._before_counts), max(self._after_counts)) or 1
        bar_width = 30
        lines: list[str] = []

        for i in range(self._num_bins):
            edge_lo = self._edges[i]
            bc = self._before_counts[i]
            ac = self._after_counts[i]

            # Before: dim, After: bright
            b_frac = bc / max_count
            a_frac = ac / max_count

            b_len = int(b_frac * bar_width)
            a_len = int(a_frac * bar_width)

            b_bar = _BLOCK_CHARS[8] * b_len
            a_bar = _BLOCK_CHARS[8] * a_len

            label = f"{edge_lo:+.3f}"
            lines.append(
                f"[{COLORS['text_muted']}]{label:>8}[/] "
                f"[dim #6a7a8a]{b_bar}[/]"
            )
            lines.append(
                f"{'':>8} "
                f"[bold {COLORS['accent_cyan']}]{a_bar}[/]"
            )

        lines.append("")
        lines.append(
            f"  [dim #6a7a8a]{_BLOCK_CHARS[8]*2}[/] before  "
            f"[bold {COLORS['accent_cyan']}]{_BLOCK_CHARS[8]*2}[/] after"
        )
        return "\n".join(lines)

    def _render_stats(self) -> str:
        """Build single-line Rich-markup summary of distribution statistics."""
        return (
            f"mean: [bold]{self._mean:+.4f}[/]  "
            f"std: [bold]{self._std:.4f}[/]  "
            f"range: [bold][{self._min_val:+.4f}, {self._max_val:+.4f}][/]  "
            f"sparsity: [bold]{self._sparsity:.1f}%[/]"
        )
