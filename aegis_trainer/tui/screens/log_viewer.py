"""
LogViewerScreen — Scrollable, filterable log output viewer.

Features:
  - Scrollable log display with Rich markup
  - Filter by level: F key cycles through DEBUG / INFO / WARN / ERROR
  - Search: / key opens search input
  - Follow mode: G key jumps to the latest entry (bottom)
  - Auto-scroll when in follow mode

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
from collections import deque

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, Label, RichLog, Static

from aegis_trainer.tui.theme import COLORS

logger = logging.getLogger(__name__)


# Log level display order for cycling
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]

LEVEL_COLORS = {
    "DEBUG": COLORS["text_muted"],
    "INFO": COLORS["accent_cyan"],
    "WARNING": COLORS["accent_yellow"],
    "ERROR": COLORS["accent_red"],
    "CRITICAL": COLORS["accent_red"],
}


class LogViewerScreen(Widget):
    """Scrollable log viewer with filtering and search."""

    BINDINGS = [
        Binding("f", "cycle_filter", "Filter Level"),
        Binding("/", "open_search", "Search"),
        Binding("g", "follow", "Follow/Bottom"),
        Binding("escape", "close_search", "Close Search", show=False),
    ]

    DEFAULT_CSS = """
    LogViewerScreen {
        height: 1fr;
        layout: vertical;
    }
    LogViewerScreen .lv-toolbar {
        height: 3;
        layout: horizontal;
        background: #131924;
        padding: 1;
    }
    LogViewerScreen .lv-filter-label {
        width: auto;
        padding: 0 2 0 0;
    }
    LogViewerScreen .lv-search-input {
        width: 1fr;
        display: none;
    }
    LogViewerScreen .lv-search-visible {
        display: block;
    }
    LogViewerScreen .lv-stats {
        width: auto;
        color: #6a7a8a;
        padding: 0 0 0 2;
    }
    LogViewerScreen RichLog {
        height: 1fr;
        background: #0a0e14;
    }
    LogViewerScreen .lv-title {
        color: #00d4ff;
        text-style: bold;
        padding: 1 0 0 1;
    }
    LogViewerScreen .lv-help {
        height: auto;
        color: #6a7a8a;
        padding: 0 1;
    }
    """

    filter_level: reactive[str] = reactive("DEBUG")
    search_visible: reactive[bool] = reactive(False)
    follow_mode: reactive[bool] = reactive(True)

    def __init__(
        self,
        max_entries: int = 10000,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._search_term: str = ""

    def compose(self) -> ComposeResult:
        yield Static("Log Viewer", classes="lv-title")
        yield Static(
            "F: Filter level  |  /: Search  |  G: Follow/Bottom  |  Esc: Close search",
            classes="lv-help",
        )

        with Horizontal(classes="lv-toolbar"):
            yield Label("", id="lv-filter-label", classes="lv-filter-label")
            yield Input(
                placeholder="Search logs...",
                id="lv-search-input",
                classes="lv-search-input",
            )
            yield Static("", id="lv-stats", classes="lv-stats")

        yield RichLog(highlight=True, markup=True, id="lv-log")

    def on_mount(self) -> None:
        """Set initial filter display."""
        self._update_filter_label()
        self._update_stats()

    def watch_filter_level(self, _value: str) -> None:
        self._update_filter_label()
        self._rerender_log()

    def watch_search_visible(self, value: bool) -> None:
        try:
            search_input = self.query_one("#lv-search-input", Input)
            if value:
                search_input.add_class("lv-search-visible")
                search_input.remove_class("lv-search-input")
                search_input.focus()
            else:
                search_input.add_class("lv-search-input")
                search_input.remove_class("lv-search-visible")
                self._search_term = ""
                self._rerender_log()
        except Exception:
            pass

    def _update_filter_label(self) -> None:
        """Update the filter level indicator."""
        try:
            label = self.query_one("#lv-filter-label", Label)
            color = LEVEL_COLORS.get(self.filter_level, COLORS["text"])
            label.update(f"Level: [{color}]{self.filter_level}+[/]")
        except Exception:
            pass

    def _update_stats(self) -> None:
        """Update the entry count display."""
        try:
            stats = self.query_one("#lv-stats", Static)
            total = len(self._entries)
            visible = sum(1 for e in self._entries if self._matches(e))
            follow_str = " [FOLLOW]" if self.follow_mode else ""
            stats.update(f"{visible}/{total} entries{follow_str}")
        except Exception:
            pass

    def _matches(self, entry: dict) -> bool:
        """Check if an entry passes current filter and search criteria."""
        # Level filter
        entry_level = entry.get("level", "INFO")
        min_index = LOG_LEVELS.index(self.filter_level) if self.filter_level in LOG_LEVELS else 0
        entry_index = LOG_LEVELS.index(entry_level) if entry_level in LOG_LEVELS else 0
        if entry_index < min_index:
            return False

        # Search filter
        if self._search_term:
            message = entry.get("message", "")
            if self._search_term.lower() not in message.lower():
                return False

        return True

    def _rerender_log(self) -> None:
        """Re-render the full log with current filters applied."""
        try:
            log_widget = self.query_one("#lv-log", RichLog)
        except Exception:
            return

        log_widget.clear()
        for entry in self._entries:
            if self._matches(entry):
                log_widget.write(self._format_entry(entry))

        self._update_stats()

    @staticmethod
    def _format_entry(entry: dict) -> str:
        """Format a single log entry with Rich markup."""
        level = entry.get("level", "INFO")
        timestamp = entry.get("timestamp", "")
        source = entry.get("source", "")
        message = entry.get("message", "")

        color = LEVEL_COLORS.get(level, COLORS["text"])
        ts_str = f"[{COLORS['text_muted']}]{timestamp}[/] " if timestamp else ""
        src_str = f"[{COLORS['text_muted']}]{source}[/] " if source else ""
        level_str = f"[{color}]{level:>7}[/]"

        return f"{ts_str}{level_str} {src_str}{message}"

    def add_entry(
        self,
        message: str,
        level: str = "INFO",
        source: str = "",
        timestamp: str = "",
    ) -> None:
        """Append a log entry and display it if it matches filters.

        Args:
            message: Log message text.
            level: Log level string (DEBUG, INFO, WARNING, ERROR).
            source: Source module/component name.
            timestamp: Formatted timestamp string.
        """
        entry = {
            "message": message,
            "level": level.upper(),
            "source": source,
            "timestamp": timestamp,
        }
        self._entries.append(entry)

        if self._matches(entry):
            try:
                log_widget = self.query_one("#lv-log", RichLog)
                log_widget.write(self._format_entry(entry))
            except Exception:
                pass

        self._update_stats()

    def action_cycle_filter(self) -> None:
        """Cycle through log filter levels."""
        try:
            current_idx = LOG_LEVELS.index(self.filter_level)
        except ValueError:
            current_idx = 0
        next_idx = (current_idx + 1) % len(LOG_LEVELS)
        self.filter_level = LOG_LEVELS[next_idx]

    def action_open_search(self) -> None:
        """Toggle search input visibility."""
        self.search_visible = not self.search_visible

    def action_close_search(self) -> None:
        """Close search input."""
        self.search_visible = False

    def action_follow(self) -> None:
        """Toggle follow mode and scroll to bottom."""
        self.follow_mode = not self.follow_mode
        if self.follow_mode:
            try:
                log_widget = self.query_one("#lv-log", RichLog)
                log_widget.scroll_end(animate=False)
            except Exception:
                pass
        self._update_stats()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle search input submission."""
        if event.input.id == "lv-search-input":
            self._search_term = event.value.strip()
            self._rerender_log()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Live search as user types."""
        if event.input.id == "lv-search-input":
            self._search_term = event.value.strip()
            self._rerender_log()
