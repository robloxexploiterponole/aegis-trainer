"""
WeightVisualizerScreen -- Live weight visualization during model operations.

Combines the Weight Atlas (braille scatter plot), Weight Histogram
(distribution chart), and Heretic-style progress display into a single
screen that updates in real-time as the trainer processes each layer.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static

from aegis_trainer.layer_context import LayerContext, _get_layer_type
from aegis_trainer.tui.theme import COLORS, LAYER_COLORS
from aegis_trainer.tui.widgets.layer_map import LayerMap

# Lazy imports for widgets that other agents are building in parallel.
# Handle ImportError gracefully so the screen can still load in a degraded
# state even before the sibling widgets are merged.

try:
    from aegis_trainer.tui.widgets.weight_atlas import WeightAtlas

    _HAS_WEIGHT_ATLAS = True
except ImportError:
    _HAS_WEIGHT_ATLAS = False

try:
    from aegis_trainer.tui.widgets.weight_histogram import WeightHistogram

    _HAS_WEIGHT_HISTOGRAM = True
except ImportError:
    _HAS_WEIGHT_HISTOGRAM = False

try:
    from aegis_trainer.tui.widgets.heretic_progress import HereticProgress

    _HAS_HERETIC_PROGRESS = True
except ImportError:
    _HAS_HERETIC_PROGRESS = False

# Optional torch dependency (needed for weight inspection)
try:
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

logger = logging.getLogger(__name__)

# Total layers in the Qwen3-Next architecture
_TOTAL_LAYERS = 48


class WeightVisualizerScreen(Widget):
    """Live weight visualization widget used as tab content.

    Layout::

        +--------------------------------------------------+
        |  AEGIS Weight Visualizer -- Layer 12 / 47        |
        |  [DeltaNet . linear_attention]                   |
        +-------------------------+------------------------+
        |                         |  Distribution          |
        |    Weight Atlas         |  (histogram)           |
        |    (braille scatter)    |  mean: +0.0012         |
        |                         |  std:   0.0341         |
        |                         |  range: [-0.18, +0.21] |
        |                         +------------------------+
        |                         |  Operation Progress    |
        |                         |  * Loading layer 12... |
        |                         |  Elapsed: 02:34        |
        +-------------------------+------------------------+
        | Layer Map: [grid of 48 cells]                    |
        | Progress bar                                     |
        +--------------------------------------------------+
    """

    BINDINGS = [
        Binding("left", "prev_layer", "Prev Layer"),
        Binding("right", "next_layer", "Next Layer"),
        Binding("home", "first_layer", "First Layer"),
        Binding("end", "last_layer", "Last Layer"),
    ]

    DEFAULT_CSS = """
    WeightVisualizerScreen {
        height: 1fr;
        layout: vertical;
    }
    WeightVisualizerScreen .wv-header {
        height: auto;
        padding: 1 1 0 1;
    }
    WeightVisualizerScreen .wv-title {
        color: #00d4ff;
        text-style: bold;
    }
    WeightVisualizerScreen .wv-subtitle {
        color: #6a7a8a;
        padding: 0 0 1 0;
    }
    WeightVisualizerScreen .wv-body {
        height: 1fr;
        layout: horizontal;
    }
    WeightVisualizerScreen .wv-left {
        width: 2fr;
        height: 1fr;
        border-right: solid #2a3a4a;
        padding: 1;
    }
    WeightVisualizerScreen .wv-right {
        width: 1fr;
        height: 1fr;
        layout: vertical;
    }
    WeightVisualizerScreen .wv-histogram-area {
        height: 1fr;
        border-bottom: solid #2a3a4a;
        padding: 1;
    }
    WeightVisualizerScreen .wv-progress-area {
        height: auto;
        min-height: 8;
        padding: 1;
    }
    WeightVisualizerScreen .wv-bottom {
        height: auto;
        padding: 0 1 1 1;
        border-top: solid #2a3a4a;
    }
    WeightVisualizerScreen .wv-atlas-placeholder {
        height: 1fr;
        content-align: center middle;
        color: #6a7a8a;
        text-style: italic;
        background: #131924;
        border: solid #2a3a4a;
    }
    WeightVisualizerScreen .wv-histogram-placeholder {
        height: 1fr;
        content-align: center middle;
        color: #6a7a8a;
        text-style: italic;
        background: #131924;
        border: solid #2a3a4a;
    }
    WeightVisualizerScreen .wv-progress-placeholder {
        height: auto;
        content-align: center middle;
        color: #6a7a8a;
        text-style: italic;
        background: #131924;
        border: solid #2a3a4a;
    }
    """

    selected_layer: reactive[int] = reactive(0)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._total_layers: int = _TOTAL_LAYERS
        self._weight_cache: dict[int, dict[str, Any]] = {}
        self._phase_cache: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        # Header
        with Vertical(classes="wv-header"):
            yield Static(
                self._build_title_text(0),
                id="wv-title",
                classes="wv-title",
            )
            yield Static(
                self._build_subtitle_text(0),
                id="wv-subtitle",
                classes="wv-subtitle",
            )

        # Main body: atlas (left) + histogram/progress (right)
        with Horizontal(classes="wv-body"):
            with Vertical(classes="wv-left"):
                if _HAS_WEIGHT_ATLAS:
                    yield WeightAtlas(id="wv-atlas")
                else:
                    yield Static(
                        "Load a model to visualize weights\n\n"
                        "(WeightAtlas widget not yet available)",
                        id="wv-atlas-placeholder",
                        classes="wv-atlas-placeholder",
                    )

            with Vertical(classes="wv-right"):
                with Vertical(classes="wv-histogram-area"):
                    if _HAS_WEIGHT_HISTOGRAM:
                        yield WeightHistogram(id="wv-histogram")
                    else:
                        yield Static(
                            "Distribution\n\n"
                            "(WeightHistogram widget not yet available)",
                            id="wv-histogram-placeholder",
                            classes="wv-histogram-placeholder",
                        )

                with Vertical(classes="wv-progress-area"):
                    if _HAS_HERETIC_PROGRESS:
                        yield HereticProgress(id="wv-progress")
                    else:
                        yield Static(
                            "Operation Progress\n\n"
                            "Waiting for operation to start...",
                            id="wv-progress-placeholder",
                            classes="wv-progress-placeholder",
                        )

        # Bottom bar: layer map
        with Vertical(classes="wv-bottom"):
            yield LayerMap(id="wv-layer-map")

    # ------------------------------------------------------------------
    # Title / subtitle helpers
    # ------------------------------------------------------------------

    def _build_title_text(self, layer_idx: int) -> str:
        """Build the header title string for a given layer index."""
        return (
            f"AEGIS Weight Visualizer -- Layer {layer_idx} / "
            f"{self._total_layers - 1}"
        )

    def _build_subtitle_text(self, layer_idx: int) -> str:
        """Build the subtitle string showing layer type."""
        layer_type = _get_layer_type(layer_idx)
        is_deltanet = layer_type == "linear_attention"
        type_label = "DeltaNet" if is_deltanet else "Attention"
        color = LAYER_COLORS.get(layer_type, COLORS["text_muted"])
        return f"[{color}][{type_label} . {layer_type}][/]"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Initialize the layer map with the default Qwen3-Next pattern."""
        # Layer map is already initialized with the default pattern via
        # LayerMap's own constructor. Highlight layer 0.
        try:
            layer_map = self.query_one("#wv-layer-map", LayerMap)
            layer_map.update_layer(0, "processing")
        except Exception:
            pass

        # Display placeholder text in the atlas area
        if _HAS_WEIGHT_ATLAS:
            try:
                atlas = self.query_one("#wv-atlas", WeightAtlas)
                if hasattr(atlas, "set_placeholder"):
                    atlas.set_placeholder("Load a model to visualize weights")
            except Exception:
                pass

        logger.info("WeightVisualizerScreen mounted")

    # ------------------------------------------------------------------
    # Layer navigation (reactive watcher + actions)
    # ------------------------------------------------------------------

    def watch_selected_layer(self, value: int) -> None:
        """Called automatically when selected_layer changes."""
        self._refresh_header(value)
        self._refresh_layer_map(value)
        self._refresh_from_cache(value)

    def action_prev_layer(self) -> None:
        """Move to the previous layer."""
        if self.selected_layer > 0:
            self.selected_layer -= 1

    def action_next_layer(self) -> None:
        """Move to the next layer."""
        if self.selected_layer < self._total_layers - 1:
            self.selected_layer += 1

    def action_first_layer(self) -> None:
        """Jump to layer 0."""
        self.selected_layer = 0

    def action_last_layer(self) -> None:
        """Jump to the last layer."""
        self.selected_layer = self._total_layers - 1

    # ------------------------------------------------------------------
    # Internal refresh helpers
    # ------------------------------------------------------------------

    def _refresh_header(self, layer_idx: int) -> None:
        """Update header title and subtitle for the given layer index."""
        try:
            title = self.query_one("#wv-title", Static)
            title.update(self._build_title_text(layer_idx))
        except Exception:
            pass

        try:
            subtitle = self.query_one("#wv-subtitle", Static)
            subtitle.update(self._build_subtitle_text(layer_idx))
        except Exception:
            pass

    def _refresh_layer_map(self, layer_idx: int) -> None:
        """Update the layer map to highlight the selected layer."""
        try:
            layer_map = self.query_one("#wv-layer-map", LayerMap)
            # Reset all non-completed layers to pending, then highlight current
            for i in range(self._total_layers):
                current_status = layer_map._statuses[i] if i < len(layer_map._statuses) else "pending"
                if current_status not in ("completed", "error"):
                    layer_map.update_layer(i, "pending")
            layer_map.update_layer(layer_idx, "processing")
        except Exception:
            pass

    def _refresh_from_cache(self, layer_idx: int) -> None:
        """If weight data is cached for this layer, push it to the widgets."""
        if layer_idx not in self._weight_cache:
            return

        cached = self._weight_cache[layer_idx]
        state_dict = cached.get("state_dict")
        ctx = cached.get("ctx")
        phase = cached.get("phase", "before")

        if state_dict is None:
            return

        if phase == "before":
            self._update_atlas_weights(state_dict, ctx)
            self._update_histogram_distribution(state_dict, ctx)
        elif phase == "after":
            self._update_atlas_modified(state_dict, ctx)
            self._update_histogram_comparison(state_dict, ctx)

    # ------------------------------------------------------------------
    # Atlas update helpers
    # ------------------------------------------------------------------

    def _update_atlas_weights(self, state_dict: dict, ctx: Any) -> None:
        """Push original weights to the atlas widget."""
        if not _HAS_WEIGHT_ATLAS:
            return
        try:
            atlas = self.query_one("#wv-atlas", WeightAtlas)
            if hasattr(atlas, "update_weights"):
                atlas.update_weights(state_dict, ctx)
        except Exception as exc:
            logger.debug("Atlas update_weights error: %s", exc)

    def _update_atlas_modified(self, state_dict: dict, ctx: Any) -> None:
        """Push modified weights to the atlas widget for diff overlay."""
        if not _HAS_WEIGHT_ATLAS:
            return
        try:
            atlas = self.query_one("#wv-atlas", WeightAtlas)
            if hasattr(atlas, "update_modified"):
                atlas.update_modified(state_dict, ctx)
        except Exception as exc:
            logger.debug("Atlas update_modified error: %s", exc)

    # ------------------------------------------------------------------
    # Histogram update helpers
    # ------------------------------------------------------------------

    def _update_histogram_distribution(self, state_dict: dict, ctx: Any) -> None:
        """Push weight distribution to the histogram widget."""
        if not _HAS_WEIGHT_HISTOGRAM:
            return
        try:
            histogram = self.query_one("#wv-histogram", WeightHistogram)
            if hasattr(histogram, "update_distribution"):
                histogram.update_distribution(state_dict, ctx)
        except Exception as exc:
            logger.debug("Histogram update_distribution error: %s", exc)

    def _update_histogram_comparison(self, state_dict: dict, ctx: Any) -> None:
        """Switch the histogram to comparison mode showing before/after."""
        if not _HAS_WEIGHT_HISTOGRAM:
            return
        try:
            histogram = self.query_one("#wv-histogram", WeightHistogram)
            if hasattr(histogram, "set_comparison_mode"):
                histogram.set_comparison_mode(state_dict, ctx)
        except Exception as exc:
            logger.debug("Histogram set_comparison_mode error: %s", exc)

    # ------------------------------------------------------------------
    # Public API -- called by the TUI app
    # ------------------------------------------------------------------

    def receive_weights(
        self,
        state_dict: dict,
        ctx: LayerContext | None = None,
        phase: str = "before",
    ) -> None:
        """Receive weight data from the trainer and update widgets.

        Called by the TUI app when the trainer sends weight data.

        Args:
            state_dict: Layer state dict mapping tensor names to values.
            ctx: LayerContext for the current layer, or None.
            phase: ``"before"`` for original weights, ``"after"`` for
                modified weights.
        """
        layer_idx = ctx.layer_index if ctx is not None else self.selected_layer

        # Cache for later if user navigates back to this layer
        self._weight_cache[layer_idx] = {
            "state_dict": state_dict,
            "ctx": ctx,
            "phase": phase,
        }
        self._phase_cache[layer_idx] = phase

        # Auto-navigate to the layer being processed
        if self.selected_layer != layer_idx:
            self.selected_layer = layer_idx
        else:
            # Already on this layer, refresh directly
            self._refresh_from_cache(layer_idx)

        logger.debug(
            "Received weights for layer %d (phase=%s, keys=%d)",
            layer_idx,
            phase,
            len(state_dict),
        )

    def receive_progress(self, progress: Any) -> None:
        """Receive a progress update from the trainer.

        Called by the TUI app when the trainer sends a ProgressUpdate.
        Forwards the update to the HereticProgress widget and updates
        the layer map status.

        Args:
            progress: A ProgressUpdate dataclass instance.
        """
        # Forward to heretic progress widget
        if _HAS_HERETIC_PROGRESS:
            try:
                heretic = self.query_one("#wv-progress", HereticProgress)
                if hasattr(heretic, "update_progress"):
                    heretic.update_progress(progress)
            except Exception as exc:
                logger.debug("HereticProgress update error: %s", exc)
        else:
            # Fallback: update the placeholder static
            try:
                placeholder = self.query_one("#wv-progress-placeholder", Static)
                op_type = getattr(progress, "operation_type", "?")
                substep = getattr(progress, "substep", "?")
                current = getattr(progress, "current_layer", 0)
                total = getattr(progress, "total_layers", self._total_layers)
                elapsed = getattr(progress, "elapsed_seconds", 0.0)
                minutes = int(elapsed) // 60
                seconds = int(elapsed) % 60
                placeholder.update(
                    f"Operation Progress\n"
                    f"  {op_type}: {substep}\n"
                    f"  Layer {current + 1}/{total}\n"
                    f"  Elapsed: {minutes:02d}:{seconds:02d}"
                )
            except Exception:
                pass

        # Update layer map status based on progress
        current_layer = getattr(progress, "current_layer", None)
        substep = getattr(progress, "substep", "")
        total = getattr(progress, "total_layers", self._total_layers)

        if current_layer is not None:
            self._total_layers = total
            try:
                layer_map = self.query_one("#wv-layer-map", LayerMap)
                # Mark all previous layers as completed
                if current_layer > 0:
                    layer_map.set_completed_up_to(current_layer - 1)
                # Mark current layer as processing
                layer_map.update_layer(current_layer, "processing")
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def total_layers(self) -> int:
        """Return the total number of model layers."""
        return self._total_layers

    @total_layers.setter
    def total_layers(self, value: int) -> None:
        """Update the total layer count."""
        self._total_layers = value
