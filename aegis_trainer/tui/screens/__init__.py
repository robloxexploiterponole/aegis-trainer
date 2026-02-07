"""
TUI screen exports for AEGIS AI Trainer.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from aegis_trainer.tui.screens.dashboard import DashboardScreen
from aegis_trainer.tui.screens.model_browser import ModelBrowserScreen
from aegis_trainer.tui.screens.operation_builder import OperationBuilderScreen
from aegis_trainer.tui.screens.layer_inspector import LayerInspectorScreen
from aegis_trainer.tui.screens.queue_manager import QueueManagerScreen
from aegis_trainer.tui.screens.log_viewer import LogViewerScreen
from aegis_trainer.tui.screens.weight_visualizer import WeightVisualizerScreen

__all__ = [
    "DashboardScreen",
    "ModelBrowserScreen",
    "OperationBuilderScreen",
    "LayerInspectorScreen",
    "QueueManagerScreen",
    "LogViewerScreen",
    "WeightVisualizerScreen",
]
