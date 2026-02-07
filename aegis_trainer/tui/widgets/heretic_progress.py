"""
HereticProgress -- Step-by-step operation progress display.

Modeled after Heretic's CLI output style: Rich-formatted step-by-step
progress with colored bullets, elapsed/ETA timing, and memory usage.

Example output::

    * Loading layer 12 (DeltaNet)...         [OK]
    * Abliterating 514 target tensors...     [OK]
    * Validating weight norms...             [OK]
    * Saving to output directory...          [OK]

    Elapsed: 02:34  |  ETA: 08:12  |  4.2 layers/min
    RAM: 45.2 / 120 GB (37.7%)  VRAM: 8.1 / 11 GB (73.6%)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import RichLog, Static

from aegis_trainer.tui.theme import COLORS, resource_color

if TYPE_CHECKING:
    from aegis_trainer.layer_trainer import ProgressUpdate

# -----------------------------------------------------------------------
# Status rendering constants
# -----------------------------------------------------------------------

_STATUS_MAP = {
    "running": (COLORS["accent_yellow"], "\u25cf", "..."),
    "ok":      (COLORS["accent_green"],  "\u25cf", "OK"),
    "warn":    (COLORS["accent_yellow"], "\u25cf", "WARN"),
    "error":   (COLORS["accent_red"],    "\u25cf", "FAIL"),
    "skip":    (COLORS["text_muted"],    "\u2500", "SKIP"),
}


@dataclass
class _Step:
    """Internal bookkeeping for a single progress step."""

    message: str
    status: str = "running"
    detail: str = ""
    line_index: int = -1


class HereticProgress(Widget):
    """Heretic-style step-by-step operation progress panel.

    Call :meth:`start_operation` to begin, :meth:`add_step` /
    :meth:`complete_step` for individual lines, or feed
    :py:class:`~aegis_trainer.layer_trainer.ProgressUpdate` objects
    to :meth:`update_progress` for automatic step management.
    """

    DEFAULT_CSS = """
    HereticProgress {
        height: auto;
        min-height: 8;
        background: #131924;
        padding: 1;
    }
    HereticProgress #hp-header {
        color: #00d4ff;
        text-style: bold;
        padding: 0 0 1 0;
    }
    HereticProgress #hp-log {
        height: auto;
        max-height: 20;
        background: #131924;
        border: none;
        padding: 0;
    }
    HereticProgress #hp-timing {
        color: #6a7a8a;
        padding: 1 0 0 0;
    }
    HereticProgress #hp-resources {
        color: #6a7a8a;
    }
    """

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._steps: list[_Step] = []
        self._operation_name: str = ""
        self._total_layers: int = 0
        self._last_substep: str = ""
        self._last_layer: int = -1

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("", id="hp-header")
        yield RichLog(highlight=True, markup=True, id="hp-log")
        yield Static("", id="hp-timing")
        yield Static("", id="hp-resources")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_operation(self, name: str, total_layers: int) -> None:
        """Begin a new operation, resetting the display.

        Sets the header banner and clears all previous steps.

        Args:
            name: Operation name (e.g. "abliteration").
            total_layers: Total layer count for progress display.
        """
        self._operation_name = name
        self._total_layers = total_layers
        self._steps.clear()
        self._last_substep = ""
        self._last_layer = -1

        try:
            header = self.query_one("#hp-header", Static)
            header.update(
                f"[bold {COLORS['accent_cyan']}]AEGIS[/] \u2014 {name}  "
                f"[{COLORS['text_muted']}]{total_layers} layers[/]"
            )
        except Exception:
            pass

        try:
            self.query_one("#hp-log", RichLog).clear()
        except Exception:
            pass

        try:
            self.query_one("#hp-timing", Static).update("")
        except Exception:
            pass

        try:
            self.query_one("#hp-resources", Static).update("")
        except Exception:
            pass

    def add_step(self, message: str, status: str = "running") -> None:
        """Append a new step line to the log.

        Args:
            message: Description of the step.
            status: One of ``"running"``, ``"ok"``, ``"warn"``,
                ``"error"``, ``"skip"``.
        """
        step = _Step(message=message, status=status)
        self._steps.append(step)
        self._write_step(step)

    def complete_step(self, status: str = "ok", detail: str = "") -> None:
        """Mark the most recent running step as completed.

        Args:
            status: Final status (``"ok"``, ``"warn"``, ``"error"``, etc.).
            detail: Optional detail text appended after the status tag.
        """
        # Find the last step that is still running
        for step in reversed(self._steps):
            if step.status == "running":
                step.status = status
                step.detail = detail
                self._rewrite_step(step)
                return

    def update_progress(self, progress: ProgressUpdate) -> None:
        """Process a :class:`ProgressUpdate` dataclass.

        Automatically creates/completes steps based on the ``substep``
        field and refreshes the timing and resource displays.

        Args:
            progress: A ``ProgressUpdate`` from the layer trainer.
        """
        layer_changed = progress.current_layer != self._last_layer
        substep_changed = progress.substep != self._last_substep

        # Complete previous running step if layer or substep changed
        if (layer_changed or substep_changed) and self._last_substep:
            self.complete_step("ok")

        # Auto-generate step text
        if substep_changed or layer_changed:
            msg = self._substep_message(progress)
            if msg:
                self.add_step(msg)

        self._last_layer = progress.current_layer
        self._last_substep = progress.substep

        # Update timing and resources
        try:
            self.query_one("#hp-timing", Static).update(
                self._render_timing(progress)
            )
        except Exception:
            pass

        try:
            self.query_one("#hp-resources", Static).update(
                self._render_resources(progress)
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering helpers
    # ------------------------------------------------------------------

    def _write_step(self, step: _Step) -> None:
        """Write a step line to the RichLog."""
        try:
            log = self.query_one("#hp-log", RichLog)
        except Exception:
            return

        line = self._format_step(step)
        step.line_index = len(self._steps) - 1
        log.write(line)

    def _rewrite_step(self, step: _Step) -> None:
        """Re-render the log from scratch after a step update.

        RichLog does not support in-place line edits, so we clear and
        replay all steps.
        """
        try:
            log = self.query_one("#hp-log", RichLog)
        except Exception:
            return

        log.clear()
        for s in self._steps:
            log.write(self._format_step(s))

    @staticmethod
    def _format_step(step: _Step) -> str:
        """Format a single step as Rich markup."""
        color, bullet, tag = _STATUS_MAP.get(
            step.status, _STATUS_MAP["running"]
        )

        if step.status == "running":
            # Running: bullet + message, no tag yet
            return f"  [{color}]{bullet}[/] {step.message}"

        detail_part = f"  {step.detail}" if step.detail else ""
        return (
            f"  [{color}]{bullet}[/] {step.message}"
            f"  [[bold {color}]{tag}[/]]{detail_part}"
        )

    @staticmethod
    def _substep_message(progress: ProgressUpdate) -> str:
        """Generate a user-facing step message from a ProgressUpdate."""
        layer_label = (
            f"layer {progress.current_layer + 1} ({progress.layer_type})"
        )
        substep = progress.substep.lower()
        if substep == "loading":
            return f"Loading {layer_label}..."
        if substep == "modifying":
            return f"Applying {progress.operation_type} to {layer_label}..."
        if substep == "saving":
            return "Saving modified weights..."
        if substep == "verifying":
            return "Validating weight integrity..."
        # Fallback
        return f"{progress.substep} -- {layer_label}..."

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 0:
            return "--:--"
        total = int(seconds)
        h, remainder = divmod(total, 3600)
        m, s = divmod(remainder, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def _render_timing(self, progress: ProgressUpdate) -> str:
        """Produce the timing summary line."""
        elapsed = self._format_time(progress.elapsed_seconds)
        eta = self._format_time(progress.eta_seconds)
        rate = progress.layers_per_minute
        layer_str = (
            f"Layer {progress.current_layer + 1}/{progress.total_layers}"
        )
        return (
            f"Elapsed: {elapsed}  |  ETA: {eta}  |  "
            f"{rate:.1f} layers/min  |  {layer_str}"
        )

    @staticmethod
    def _render_resources(progress: ProgressUpdate) -> str:
        """Produce the resource usage line with color coding."""
        # RAM
        ram_pct = (
            progress.ram_used_gb / progress.ram_total_gb * 100.0
            if progress.ram_total_gb > 0
            else 0.0
        )
        ram_color = resource_color(ram_pct)

        # VRAM
        vram_pct = (
            progress.vram_used_gb / progress.vram_total_gb * 100.0
            if progress.vram_total_gb > 0
            else 0.0
        )
        vram_color = resource_color(vram_pct)

        return (
            f"RAM: [{ram_color}]{progress.ram_used_gb:.1f}[/] / "
            f"{progress.ram_total_gb:.0f} GB ({ram_pct:.1f}%)  "
            f"VRAM: [{vram_color}]{progress.vram_used_gb:.1f}[/] / "
            f"{progress.vram_total_gb:.0f} GB ({vram_pct:.1f}%)"
        )
