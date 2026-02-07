"""
AegisTrainerApp — Main Textual application for the AEGIS AI Trainer TUI.

Provides a tabbed interface with seven panes:
  1. Dashboard   — Resource monitoring, progress, queue summary, log tail
  2. Models      — Browse and inspect available models
  3. Build       — Step-by-step operation configuration wizard
  4. Inspect     — Per-layer weight inspector with arrow-key navigation
  5. Queue       — Queue management (reorder, cancel, pause/resume)
  6. Logs        — Scrollable, filterable log viewer
  7. Visualizer  — Live weight visualization (atlas, histogram, progress)

Keyboard shortcuts:
  1-7        Switch to corresponding tab
  F1         Help
  F10        Quit

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, TabbedContent, TabPane

from aegis_trainer.tui.theme import AEGIS_CSS
from aegis_trainer.tui.screens.dashboard import DashboardScreen
from aegis_trainer.tui.screens.model_browser import ModelBrowserScreen
from aegis_trainer.tui.screens.operation_builder import OperationBuilderScreen
from aegis_trainer.tui.screens.layer_inspector import LayerInspectorScreen
from aegis_trainer.tui.screens.queue_manager import QueueManagerScreen
from aegis_trainer.tui.screens.log_viewer import LogViewerScreen
from aegis_trainer.tui.screens.weight_visualizer import WeightVisualizerScreen

logger = logging.getLogger(__name__)


class AegisTrainerApp(App):
    """AEGIS AI Trainer — Terminal User Interface.

    A full-featured TUI for managing layer-by-layer model training operations
    on consumer GPUs using AirLLM layer streaming.
    """

    CSS = AEGIS_CSS
    TITLE = "AEGIS AI Trainer"
    SUB_TITLE = "Hardwick Software Services"

    BINDINGS = [
        Binding("1", "switch_tab('dashboard')", "Dashboard", show=True),
        Binding("2", "switch_tab('models')", "Models", show=True),
        Binding("3", "switch_tab('build')", "Build", show=True),
        Binding("4", "switch_tab('inspect')", "Inspect", show=True),
        Binding("5", "switch_tab('queue')", "Queue", show=True),
        Binding("6", "switch_tab('logs')", "Logs", show=True),
        Binding("7", "switch_tab('tab-viz')", "Visualizer", show=True),
        Binding("f1", "help_screen", "Help"),
        Binding("f10", "quit", "Quit"),
    ]

    def __init__(self, model_path: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_path = model_path

    def compose(self) -> ComposeResult:
        yield Header()

        with TabbedContent(initial="dashboard"):
            with TabPane("Dashboard", id="dashboard"):
                yield DashboardScreen()

            with TabPane("Models", id="models"):
                yield ModelBrowserScreen()

            with TabPane("Build", id="build"):
                yield OperationBuilderScreen()

            with TabPane("Inspect", id="inspect"):
                yield LayerInspectorScreen(model_path=self._model_path)

            with TabPane("Queue", id="queue"):
                yield QueueManagerScreen()

            with TabPane("Logs", id="logs"):
                yield LogViewerScreen()

            with TabPane("Visualizer", id="tab-viz"):
                yield WeightVisualizerScreen()

        yield Footer()

    def action_switch_tab(self, tab_id: str) -> None:
        """Switch to the specified tab pane."""
        try:
            tabbed = self.query_one(TabbedContent)
            tabbed.active = tab_id
        except Exception:
            pass

    def action_help_screen(self) -> None:
        """Display a help notification with keyboard shortcuts."""
        self.notify(
            "Keyboard shortcuts:\n"
            "  1-7: Switch tabs\n"
            "  F1:  This help\n"
            "  F10: Quit\n\n"
            "Dashboard:  Live resource monitoring\n"
            "Models:     Browse /AEGIS_AI/models/\n"
            "Build:      Configure new operations\n"
            "Inspect:    Layer-by-layer weight viewer\n"
            "Queue:      Manage operation queue\n"
            "Logs:       Filtered log viewer\n"
            "Visualizer: Live weight visualization",
            title="AEGIS AI Trainer Help",
            severity="information",
        )

    def on_mount(self) -> None:
        """Log startup."""
        logger.info("AEGIS AI Trainer TUI started")
        # Push initial log entry to the dashboard
        try:
            dashboard = self.query_one(DashboardScreen)
            dashboard.append_log("[bold cyan]AEGIS AI Trainer TUI started[/]")
        except Exception:
            pass

    def notify_weights(
        self,
        state_dict: dict,
        ctx: object | None = None,
        phase: str = "before",
    ) -> None:
        """Forward weight data to the Weight Visualizer screen.

        Called by the LayerTrainer's weight_callback to push weight
        tensors into the live visualization.

        Args:
            state_dict: Layer state dict (tensor name -> value mapping).
            ctx: LayerContext for the current layer.
            phase: ``"before"`` for original weights, ``"after"`` for
                modified weights.
        """
        try:
            visualizer = self.query_one(WeightVisualizerScreen)
            visualizer.receive_weights(state_dict, ctx, phase)
        except Exception:
            pass

    def notify_progress(self, progress: object) -> None:
        """Forward a progress update to the Weight Visualizer screen.

        Called by the LayerTrainer's progress_callback to update the
        live progress display in the visualizer.

        Args:
            progress: A ProgressUpdate dataclass instance.
        """
        try:
            visualizer = self.query_one(WeightVisualizerScreen)
            visualizer.receive_progress(progress)
        except Exception:
            pass
