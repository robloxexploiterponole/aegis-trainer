"""
QueueManagerScreen — View and manage the AEGIS operation queue.

Features:
  - List queued, active, and completed jobs in a DataTable
  - Reorder queue items with U/D keys (move up/down)
  - Cancel jobs with Delete key
  - Pause/Resume with P/R keys
  - Auto-refresh every 2 seconds

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import DataTable, Label, RichLog, Static

from aegis_trainer.tui.theme import COLORS

logger = logging.getLogger(__name__)


# Job status constants
STATUS_QUEUED = "queued"
STATUS_ACTIVE = "active"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
STATUS_PAUSED = "paused"

STATUS_COLORS = {
    STATUS_QUEUED: COLORS["text_muted"],
    STATUS_ACTIVE: COLORS["accent_cyan"],
    STATUS_COMPLETED: COLORS["accent_green"],
    STATUS_FAILED: COLORS["accent_red"],
    STATUS_CANCELLED: COLORS["accent_red"],
    STATUS_PAUSED: COLORS["accent_yellow"],
}


class QueueManagerScreen(Widget):
    """Queue management screen with keyboard controls."""

    BINDINGS = [
        Binding("u", "move_up", "Move Up"),
        Binding("d", "move_down", "Move Down"),
        Binding("delete", "cancel_job", "Cancel"),
        Binding("p", "pause_job", "Pause"),
        Binding("r", "resume_job", "Resume"),
        Binding("c", "clear_completed", "Clear Done"),
    ]

    DEFAULT_CSS = """
    QueueManagerScreen {
        height: 1fr;
        layout: horizontal;
    }
    QueueManagerScreen .qm-list {
        width: 2fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    QueueManagerScreen .qm-detail {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        border-left: solid #2a3a4a;
        padding: 1;
    }
    QueueManagerScreen .qm-title {
        color: #00d4ff;
        text-style: bold;
        padding: 0 0 1 0;
    }
    QueueManagerScreen .qm-controls {
        height: auto;
        color: #6a7a8a;
        padding: 0 0 1 0;
    }
    QueueManagerScreen DataTable {
        height: 1fr;
        background: #131924;
    }
    QueueManagerScreen RichLog {
        height: 1fr;
        background: #131924;
    }
    QueueManagerScreen .qm-summary {
        height: auto;
        padding: 1 0 0 0;
        color: #6a7a8a;
    }
    """

    job_count: reactive[int] = reactive(0)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._jobs: list[dict] = []

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(classes="qm-list"):
                yield Static("Queue Manager", classes="qm-title")
                yield Static(
                    "U/D: Reorder  |  Del: Cancel  |  P: Pause  |  R: Resume  |  C: Clear completed",
                    classes="qm-controls",
                )
                yield DataTable(id="qm-job-table", cursor_type="row")
                yield Static("", id="qm-summary", classes="qm-summary")

            with Vertical(classes="qm-detail"):
                yield Static("Job Details", classes="qm-title")
                yield RichLog(highlight=True, markup=True, id="qm-detail-log")

    def on_mount(self) -> None:
        """Initialize table and start refresh timer."""
        try:
            table = self.query_one("#qm-job-table", DataTable)
            table.add_columns("#", "Operation", "Model", "Status", "Progress", "Queued")
        except Exception:
            pass

        self._refresh_jobs()
        self.set_interval(2.0, self._refresh_jobs)

    def _refresh_jobs(self) -> None:
        """Poll the queue and update the job table.

        In a full implementation this would read from the QQMS queue manager.
        Currently uses the internal job list.
        """
        self._update_summary()

    def set_jobs(self, jobs: list[dict]) -> None:
        """Replace the full job list and refresh display.

        Args:
            jobs: List of dicts with keys: id, operation, model, status, progress, queued_at.
        """
        self._jobs = list(jobs)
        self._render_table()
        self._update_summary()
        self.job_count = len(self._jobs)

    def _render_table(self) -> None:
        """Re-render the job table from self._jobs."""
        try:
            table = self.query_one("#qm-job-table", DataTable)
            table.clear()
        except Exception:
            return

        for i, job in enumerate(self._jobs):
            status = job.get("status", STATUS_QUEUED)
            progress = job.get("progress", 0.0)
            queued_at = job.get("queued_at", 0.0)

            # Format progress
            if status == STATUS_COMPLETED:
                progress_str = "100%"
            elif status in (STATUS_CANCELLED, STATUS_FAILED):
                progress_str = "-"
            else:
                progress_str = f"{progress:.0f}%"

            # Format time
            if queued_at > 0:
                elapsed = time.time() - queued_at
                time_str = _fmt_elapsed(elapsed)
            else:
                time_str = "-"

            table.add_row(
                str(i + 1),
                job.get("operation", "?"),
                _truncate(job.get("model", "?"), 30),
                status,
                progress_str,
                time_str,
            )

    def _update_summary(self) -> None:
        """Update the summary line below the table."""
        try:
            summary = self.query_one("#qm-summary", Static)
        except Exception:
            return

        total = len(self._jobs)
        queued = sum(1 for j in self._jobs if j.get("status") == STATUS_QUEUED)
        active = sum(1 for j in self._jobs if j.get("status") == STATUS_ACTIVE)
        done = sum(1 for j in self._jobs if j.get("status") == STATUS_COMPLETED)
        failed = sum(1 for j in self._jobs if j.get("status") == STATUS_FAILED)

        summary.update(
            f"Total: {total}  |  Queued: {queued}  |  "
            f"Active: {active}  |  Completed: {done}  |  Failed: {failed}"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Show details for the selected job."""
        row = event.cursor_row
        if row < 0 or row >= len(self._jobs):
            return

        job = self._jobs[row]
        try:
            detail = self.query_one("#qm-detail-log", RichLog)
        except Exception:
            return

        detail.clear()
        color = STATUS_COLORS.get(job.get("status", ""), COLORS["text"])
        detail.write(f"[bold]Job #{row + 1}[/]")
        detail.write("")
        detail.write(f"  Operation:  {job.get('operation', '?')}")
        detail.write(f"  Model:      {job.get('model', '?')}")
        detail.write(f"  Output:     {job.get('output', '?')}")
        detail.write(f"  Status:     [{color}]{job.get('status', '?')}[/]")
        detail.write(f"  Progress:   {job.get('progress', 0):.1f}%")
        detail.write("")

        params = job.get("params", {})
        if params:
            detail.write(f"  [bold]Parameters:[/]")
            for k, v in params.items():
                detail.write(f"    {k}: {v}")

    def _get_selected_index(self) -> int:
        """Get the currently highlighted row index, or -1."""
        try:
            table = self.query_one("#qm-job-table", DataTable)
            return table.cursor_row
        except Exception:
            return -1

    def action_move_up(self) -> None:
        """Move the selected job up in the queue."""
        idx = self._get_selected_index()
        if idx <= 0 or idx >= len(self._jobs):
            return
        job = self._jobs[idx]
        if job.get("status") not in (STATUS_QUEUED, STATUS_PAUSED):
            return
        self._jobs[idx], self._jobs[idx - 1] = self._jobs[idx - 1], self._jobs[idx]
        self._render_table()

    def action_move_down(self) -> None:
        """Move the selected job down in the queue."""
        idx = self._get_selected_index()
        if idx < 0 or idx >= len(self._jobs) - 1:
            return
        job = self._jobs[idx]
        if job.get("status") not in (STATUS_QUEUED, STATUS_PAUSED):
            return
        self._jobs[idx], self._jobs[idx + 1] = self._jobs[idx + 1], self._jobs[idx]
        self._render_table()

    def action_cancel_job(self) -> None:
        """Cancel the selected job."""
        idx = self._get_selected_index()
        if idx < 0 or idx >= len(self._jobs):
            return
        job = self._jobs[idx]
        if job.get("status") in (STATUS_COMPLETED, STATUS_CANCELLED):
            return
        job["status"] = STATUS_CANCELLED
        self._render_table()
        self._update_summary()
        logger.info("Cancelled job #%d: %s", idx + 1, job.get("operation"))

    def action_pause_job(self) -> None:
        """Pause the selected job."""
        idx = self._get_selected_index()
        if idx < 0 or idx >= len(self._jobs):
            return
        job = self._jobs[idx]
        if job.get("status") in (STATUS_QUEUED, STATUS_ACTIVE):
            job["status"] = STATUS_PAUSED
            self._render_table()
            self._update_summary()

    def action_resume_job(self) -> None:
        """Resume a paused job."""
        idx = self._get_selected_index()
        if idx < 0 or idx >= len(self._jobs):
            return
        job = self._jobs[idx]
        if job.get("status") == STATUS_PAUSED:
            job["status"] = STATUS_QUEUED
            self._render_table()
            self._update_summary()

    def action_clear_completed(self) -> None:
        """Remove all completed and cancelled jobs from the list."""
        self._jobs = [
            j for j in self._jobs
            if j.get("status") not in (STATUS_COMPLETED, STATUS_CANCELLED)
        ]
        self._render_table()
        self._update_summary()
        self.job_count = len(self._jobs)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time for display."""
    s = int(seconds)
    if s < 60:
        return f"{s}s ago"
    m, _ = divmod(s, 60)
    if m < 60:
        return f"{m}m ago"
    h, m = divmod(m, 60)
    return f"{h}h {m}m ago"


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
