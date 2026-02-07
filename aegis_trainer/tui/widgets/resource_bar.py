"""
ResourceBar — CPU / RAM / VRAM usage bar widget with color thresholds.

Displays a labeled progress bar that changes color based on usage level:
  0-60%  green   (#00ff88)
  60-80% yellow  (#ffd700)
  80-90% orange  (#ff8800)
  90-100% red    (#ff4444)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, ProgressBar, Static

from aegis_trainer.tui.theme import resource_color


class ResourceBar(Widget):
    """Single resource usage bar with label, bar, and percentage readout.

    Args:
        label: Display label (e.g. "CPU", "RAM", "VRAM").
        total_label: Human-readable total (e.g. "120 GB").
    """

    DEFAULT_CSS = """
    ResourceBar {
        height: 3;
        padding: 0 1;
        layout: horizontal;
    }
    ResourceBar .rb-label {
        width: 8;
        color: #6a7a8a;
        padding: 1 0;
    }
    ResourceBar ProgressBar {
        width: 1fr;
        padding: 0 1;
    }
    ResourceBar .rb-value {
        width: 22;
        color: #d4dae4;
        text-align: right;
        padding: 1 0;
    }
    """

    percent: reactive[float] = reactive(0.0)
    used_label: reactive[str] = reactive("")

    def __init__(
        self,
        label: str = "CPU",
        total_label: str = "",
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._label_text = label
        self._total_label = total_label

    def compose(self) -> ComposeResult:
        yield Label(f"{self._label_text:>6}", classes="rb-label")
        yield ProgressBar(total=100.0, show_eta=False, show_percentage=False)
        yield Static("  0.0%", classes="rb-value")

    def watch_percent(self, value: float) -> None:
        """React to percent changes — update bar and color."""
        try:
            bar = self.query_one(ProgressBar)
            bar.progress = min(value, 100.0)
            color = resource_color(value)
            bar.styles.color = color
        except Exception:
            pass

        try:
            readout = self.query_one(".rb-value", Static)
            detail = f"{value:5.1f}%"
            if self.used_label:
                detail = f"{self.used_label}  {detail}"
            readout.update(detail)
        except Exception:
            pass

    def watch_used_label(self, value: str) -> None:
        """React to label update (e.g. '45.2 / 120 GB')."""
        try:
            readout = self.query_one(".rb-value", Static)
            detail = f"{self.percent:5.1f}%"
            if value:
                detail = f"{value}  {detail}"
            readout.update(detail)
        except Exception:
            pass

    def update_value(self, percent: float, used_text: str = "") -> None:
        """Convenience: set both percent and used_label in one call."""
        self.used_label = used_text
        self.percent = percent
