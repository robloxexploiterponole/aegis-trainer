"""
LayerMap — Visual 48-cell grid showing layer types and completion status.

Each cell represents one transformer layer, colored by type:
  - DeltaNet (linear_attention): yellow (#ffd700)
  - Full Attention (RoPE/GQA):   cyan (#00d4ff)

Completion status overlays:
  - Pending:    dim / outlined
  - Processing: blinking / highlighted
  - Completed:  green (#00ff88) fill
  - Error:      red (#ff4444) fill

The default Qwen3-Next pattern is 3 DeltaNet + 1 Attention, repeated 12 times.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from aegis_trainer.tui.theme import COLORS, LAYER_COLORS

# Default Qwen3-Next layer pattern: 3 DeltaNet + 1 Attention x12
_DEFAULT_PATTERN = (["linear_attention"] * 3 + ["full_attention"]) * 12

# Layer status constants
STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_ERROR = "error"


class LayerMap(Widget):
    """Visual grid of 48 layer cells with type coloring and status overlay.

    The map is rendered as a Rich-formatted string and displayed in a Static
    widget.  Call update_layer() to change individual cell status, or
    set_pattern() to change the layer type layout.
    """

    DEFAULT_CSS = """
    LayerMap {
        height: auto;
        padding: 1;
        background: #131924;
        border: solid #2a3a4a;
    }
    LayerMap .lm-title {
        color: #00d4ff;
        text-style: bold;
        padding: 0 0 1 0;
    }
    LayerMap .lm-grid {
        padding: 0;
    }
    LayerMap .lm-legend {
        color: #6a7a8a;
        padding: 1 0 0 0;
    }
    """

    total_layers: reactive[int] = reactive(48)

    def __init__(
        self,
        layer_types: list[str] | None = None,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._layer_types: list[str] = list(layer_types or _DEFAULT_PATTERN)
        self._statuses: list[str] = [STATUS_PENDING] * len(self._layer_types)
        self._active_layer: int = -1

    def compose(self) -> ComposeResult:
        yield Static("Layer Map", classes="lm-title")
        yield Static(self._render_grid(), id="lm-grid", classes="lm-grid")
        yield Static(self._render_legend(), classes="lm-legend")

    def set_pattern(self, layer_types: list[str]) -> None:
        """Replace the layer type pattern and reset statuses."""
        self._layer_types = list(layer_types)
        self._statuses = [STATUS_PENDING] * len(self._layer_types)
        self._active_layer = -1
        self.total_layers = len(self._layer_types)
        self._refresh_grid()

    def update_layer(self, index: int, status: str) -> None:
        """Update the status of a specific layer cell.

        Args:
            index: Layer index (0-based).
            status: One of STATUS_PENDING, STATUS_PROCESSING, STATUS_COMPLETED, STATUS_ERROR.
        """
        if 0 <= index < len(self._statuses):
            self._statuses[index] = status
            if status == STATUS_PROCESSING:
                self._active_layer = index
            self._refresh_grid()

    def set_completed_up_to(self, index: int) -> None:
        """Mark all layers from 0 to index (inclusive) as completed."""
        for i in range(min(index + 1, len(self._statuses))):
            self._statuses[i] = STATUS_COMPLETED
        self._refresh_grid()

    def _refresh_grid(self) -> None:
        """Re-render and update the grid Static widget."""
        try:
            grid = self.query_one("#lm-grid", Static)
            grid.update(self._render_grid())
        except Exception:
            pass

    def _render_grid(self) -> str:
        """Build a Rich markup string representing the layer grid.

        Layout: 16 cells per row, 3 rows = 48 cells.
        Each cell is a 2-char block: type-colored if pending, green if done,
        red if error, highlighted if active.
        """
        cells_per_row = 16
        lines: list[str] = []
        idx_line: list[str] = []
        cell_line: list[str] = []

        for i, (ltype, status) in enumerate(
            zip(self._layer_types, self._statuses)
        ):
            # Determine cell color
            if status == STATUS_COMPLETED:
                color = COLORS["accent_green"]
                char = "\u2588\u2588"  # Full block
            elif status == STATUS_ERROR:
                color = COLORS["accent_red"]
                char = "XX"
            elif status == STATUS_PROCESSING:
                color = COLORS["accent_cyan"] if ltype == "full_attention" else COLORS["accent_yellow"]
                char = "\u2592\u2592"  # Medium shade (animated feel)
            else:
                # Pending — dim version of type color
                color = LAYER_COLORS.get(ltype, COLORS["text_muted"])
                char = "\u2591\u2591"  # Light shade

            cell_line.append(f"[{color}]{char}[/]")
            idx_line.append(f"[{COLORS['text_muted']}]{i:2d}[/]")

            if (i + 1) % cells_per_row == 0 or i == len(self._layer_types) - 1:
                lines.append(" ".join(idx_line))
                lines.append(" ".join(cell_line))
                lines.append("")  # spacing
                idx_line = []
                cell_line = []

        return "\n".join(lines)

    @staticmethod
    def _render_legend() -> str:
        """Build legend markup."""
        return (
            f"[{COLORS['accent_yellow']}]\u2588\u2588[/] DeltaNet  "
            f"[{COLORS['accent_cyan']}]\u2588\u2588[/] Attention  "
            f"[{COLORS['accent_green']}]\u2588\u2588[/] Done  "
            f"[{COLORS['accent_red']}]XX[/] Error  "
            f"[{COLORS['text_muted']}]\u2591\u2591[/] Pending"
        )
