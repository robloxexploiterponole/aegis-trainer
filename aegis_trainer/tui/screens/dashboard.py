"""
Dashboard screen — main landing page for the AEGIS AI Trainer TUI.

Layout:
  - ASCII art header (top center)
  - Resource monitoring bars: CPU, RAM, VRAM (2-second refresh via psutil)
  - Active operation progress panel (if running)
  - Queue sidebar showing pending jobs
  - Recent log tail (last 10 lines)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import DataTable, RichLog, Static

from aegis_trainer.tui.theme import COLORS, header_rich
from aegis_trainer.tui.widgets.progress_panel import ProgressPanel
from aegis_trainer.tui.widgets.resource_bar import ResourceBar
from aegis_trainer.utils.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)


class DashboardScreen(Widget):
    """Main dashboard combining resource bars, progress, queue, and logs."""

    DEFAULT_CSS = """
    DashboardScreen {
        height: 1fr;
        layout: vertical;
    }
    DashboardScreen .dash-header {
        height: auto;
        text-align: center;
        padding: 1;
        color: #00d4ff;
    }
    DashboardScreen .dash-resources {
        height: auto;
        padding: 0 1;
    }
    DashboardScreen .dash-main {
        height: 1fr;
        layout: horizontal;
    }
    DashboardScreen .dash-left {
        width: 2fr;
        height: 1fr;
        layout: vertical;
    }
    DashboardScreen .dash-right {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        border-left: solid #2a3a4a;
        padding: 0 1;
    }
    DashboardScreen .dash-queue-title {
        color: #00d4ff;
        text-style: bold;
        padding: 1 0 0 0;
    }
    DashboardScreen .dash-logs-title {
        color: #00d4ff;
        text-style: bold;
        padding: 1 0 0 0;
    }
    DashboardScreen DataTable {
        height: 1fr;
        background: #131924;
    }
    DashboardScreen RichLog {
        height: 12;
        background: #131924;
        border: solid #2a3a4a;
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
        self._monitor = ResourceMonitor()

    def compose(self) -> ComposeResult:
        # Header
        yield Static(header_rich(), classes="dash-header")

        # Resource bars
        with Vertical(classes="dash-resources"):
            yield ResourceBar(label="CPU", id="rb-cpu")
            yield ResourceBar(label="RAM", total_label="120 GB", id="rb-ram")
            yield ResourceBar(label="VRAM", total_label="11 GB", id="rb-vram")

        # Main area: left (progress + logs) and right (queue)
        with Horizontal(classes="dash-main"):
            with Vertical(classes="dash-left"):
                yield ProgressPanel(id="dash-progress")
                yield Static("Recent Logs", classes="dash-logs-title")
                yield RichLog(highlight=True, markup=True, id="dash-log")

            with Vertical(classes="dash-right"):
                yield Static("Queue", classes="dash-queue-title")
                yield DataTable(id="dash-queue-table")

    def on_mount(self) -> None:
        """Initialize the queue table and start the resource polling timer."""
        try:
            table = self.query_one("#dash-queue-table", DataTable)
            table.add_columns("#", "Operation", "Model", "Status")
        except Exception:
            pass

        # Populate with initial snapshot
        self._poll_resources()

        # Start 2-second refresh cycle
        self.set_interval(2.0, self._poll_resources)

    def _poll_resources(self) -> None:
        """Fetch current resource usage and update the bars."""
        try:
            snap = self._monitor.get_snapshot()
        except Exception as exc:
            logger.debug("Resource poll failed: %s", exc)
            return

        # CPU
        try:
            cpu_bar = self.query_one("#rb-cpu", ResourceBar)
            cpu_bar.update_value(snap.cpu_percent)
        except Exception:
            pass

        # RAM
        try:
            ram_bar = self.query_one("#rb-ram", ResourceBar)
            used_gb = snap.ram_used_bytes / (1024 ** 3)
            total_gb = snap.ram_total_bytes / (1024 ** 3)
            ram_bar.update_value(
                snap.ram_percent,
                f"{used_gb:.1f} / {total_gb:.1f} GB",
            )
        except Exception:
            pass

        # VRAM
        try:
            vram_bar = self.query_one("#rb-vram", ResourceBar)
            if snap.vram_total_bytes > 0:
                vused_gb = snap.vram_used_bytes / (1024 ** 3)
                vtotal_gb = snap.vram_total_bytes / (1024 ** 3)
                vram_bar.update_value(
                    snap.vram_percent,
                    f"{vused_gb:.1f} / {vtotal_gb:.1f} GB",
                )
            else:
                vram_bar.update_value(0.0, "N/A")
        except Exception:
            pass

    def append_log(self, message: str) -> None:
        """Append a line to the log tail widget."""
        try:
            log = self.query_one("#dash-log", RichLog)
            log.write(message)
        except Exception:
            pass

    def update_queue(self, items: list[dict]) -> None:
        """Refresh the queue sidebar table.

        Args:
            items: List of dicts with keys: index, operation, model, status.
        """
        try:
            table = self.query_one("#dash-queue-table", DataTable)
            table.clear()
            for item in items:
                table.add_row(
                    str(item.get("index", "?")),
                    item.get("operation", ""),
                    item.get("model", ""),
                    item.get("status", "pending"),
                )
        except Exception:
            pass
