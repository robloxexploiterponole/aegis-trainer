"""
Reusable TUI widgets for AEGIS AI Trainer.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from aegis_trainer.tui.widgets.resource_bar import ResourceBar
from aegis_trainer.tui.widgets.progress_panel import ProgressPanel
from aegis_trainer.tui.widgets.layer_map import LayerMap
from aegis_trainer.tui.widgets.braille_canvas import BrailleCanvas
from aegis_trainer.tui.widgets.weight_atlas import WeightAtlas
from aegis_trainer.tui.widgets.weight_histogram import WeightHistogram
from aegis_trainer.tui.widgets.heretic_progress import HereticProgress

__all__ = [
    "ResourceBar",
    "ProgressPanel",
    "LayerMap",
    "BrailleCanvas",
    "WeightAtlas",
    "WeightHistogram",
    "HereticProgress",
]
