"""
ProgressPanel — Displays active operation progress with per-layer substep detail.

Shows:
  - Operation name and overall percentage
  - Current layer being processed (e.g. "Layer 12/48 — linear_attention (DeltaNet)")
  - Per-substep progress within the current layer
  - Elapsed time and ETA

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static

from aegis_trainer.tui.theme import COLORS


class ProgressPanel(Widget):
    """Panel showing live operation progress with substep detail.

    Attributes:
        operation_name: Name of the active operation (e.g. "abliterate").
        current_layer: Zero-based layer index being processed.
        total_layers: Total layers in the model.
        layer_type: Current layer type string.
        substep: Current substep description.
        overall_percent: Overall completion 0-100.
    """

    DEFAULT_CSS = """
    ProgressPanel {
        height: auto;
        padding: 1;
        background: #131924;
        border: solid #2a3a4a;
    }
    ProgressPanel .pp-title {
        color: #00d4ff;
        text-style: bold;
        padding: 0 0 1 0;
    }
    ProgressPanel .pp-layer-info {
        color: #d4dae4;
        padding: 0 0 1 0;
    }
    ProgressPanel .pp-substep {
        color: #6a7a8a;
    }
    ProgressPanel .pp-timing {
        color: #6a7a8a;
        padding: 1 0 0 0;
    }
    ProgressPanel ProgressBar {
        padding: 0 0 1 0;
    }
    ProgressPanel .pp-idle {
        color: #6a7a8a;
        text-style: italic;
        padding: 1 0;
    }
    """

    operation_name: reactive[str] = reactive("")
    current_layer: reactive[int] = reactive(0)
    total_layers: reactive[int] = reactive(48)
    layer_type: reactive[str] = reactive("")
    substep: reactive[str] = reactive("")
    overall_percent: reactive[float] = reactive(0.0)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._start_time: float | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Operation Progress", classes="pp-title")
            yield Static("No active operation", classes="pp-idle", id="pp-idle")
            yield Label("", classes="pp-layer-info", id="pp-layer-info")
            yield ProgressBar(total=100.0, show_eta=False, id="pp-bar")
            yield Static("", classes="pp-substep", id="pp-substep")
            yield Static("", classes="pp-timing", id="pp-timing")

    def watch_operation_name(self, value: str) -> None:
        """Toggle between idle and active display."""
        try:
            idle = self.query_one("#pp-idle", Static)
            layer_info = self.query_one("#pp-layer-info", Label)
            bar = self.query_one("#pp-bar", ProgressBar)
            substep = self.query_one("#pp-substep", Static)
            timing = self.query_one("#pp-timing", Static)
        except Exception:
            return

        if value:
            idle.display = False
            layer_info.display = True
            bar.display = True
            substep.display = True
            timing.display = True
            self._start_time = time.time()
        else:
            idle.display = True
            idle.update("No active operation")
            layer_info.display = False
            bar.display = False
            substep.display = False
            timing.display = False
            self._start_time = None

    def watch_overall_percent(self, value: float) -> None:
        try:
            bar = self.query_one("#pp-bar", ProgressBar)
            bar.progress = min(value, 100.0)
        except Exception:
            pass
        self._update_timing()

    def watch_current_layer(self, _value: int) -> None:
        self._update_layer_info()

    def watch_layer_type(self, _value: str) -> None:
        self._update_layer_info()

    def watch_substep(self, value: str) -> None:
        try:
            self.query_one("#pp-substep", Static).update(f"  {value}")
        except Exception:
            pass

    def _update_layer_info(self) -> None:
        """Refresh the layer info label."""
        if not self.operation_name:
            return
        try:
            label = self.query_one("#pp-layer-info", Label)
        except Exception:
            return

        type_label = self.layer_type or "unknown"
        kind = "DeltaNet" if "linear" in type_label else "Attention"
        label.update(
            f"{self.operation_name}  |  Layer {self.current_layer + 1}/{self.total_layers}"
            f" - {type_label} ({kind})"
        )

    def _update_timing(self) -> None:
        """Refresh elapsed/ETA display."""
        if self._start_time is None or self.overall_percent <= 0:
            return
        try:
            timing = self.query_one("#pp-timing", Static)
        except Exception:
            return

        elapsed = time.time() - self._start_time
        if self.overall_percent > 0:
            estimated_total = elapsed / (self.overall_percent / 100.0)
            remaining = max(0.0, estimated_total - elapsed)
            timing.update(
                f"  Elapsed: {_fmt_duration(elapsed)}  |  ETA: {_fmt_duration(remaining)}"
            )
        else:
            timing.update(f"  Elapsed: {_fmt_duration(elapsed)}")

    def set_progress(
        self,
        operation: str,
        layer: int,
        total: int,
        layer_type: str,
        percent: float,
        substep: str = "",
    ) -> None:
        """Convenience: update all fields in one call."""
        self.total_layers = total
        self.operation_name = operation
        self.current_layer = layer
        self.layer_type = layer_type
        self.substep = substep
        self.overall_percent = percent

    def clear_progress(self) -> None:
        """Reset to idle state."""
        self.operation_name = ""
        self.overall_percent = 0.0
        self.current_layer = 0
        self.layer_type = ""
        self.substep = ""


def _fmt_duration(seconds: float) -> str:
    """Format seconds into Xh Ym Zs string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"
