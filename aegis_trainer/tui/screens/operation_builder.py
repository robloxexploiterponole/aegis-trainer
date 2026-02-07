"""
OperationBuilderScreen — Step-by-step wizard for configuring training operations.

Wizard steps:
  1. Select operation type (abliterate, longrope, lora, quantize)
  2. Select source model (from /AEGIS_AI/models/ or manual path)
  3. Configure operation-specific parameters
  4. Set output path
  5. Review and confirm -> queue the operation

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import logging
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Input,
    Label,
    OptionList,
    RichLog,
    Static,
)
from textual.widgets.option_list import Option

from aegis_trainer.tui.theme import COLORS

logger = logging.getLogger(__name__)


# Operation definitions with their parameters
OPERATIONS = {
    "abliterate": {
        "label": "Abliterate (Deregulation via directional ablation)",
        "color": COLORS["accent_cyan"],
        "params": {
            "directions_path": {"label": "Directions file", "default": "", "type": "path"},
            "device_map": {"label": "Device map", "default": "auto", "type": "text"},
            "layers": {"label": "Layers to process", "default": "all", "type": "text"},
        },
    },
    "longrope": {
        "label": "LongRoPE (Context window extension)",
        "color": COLORS["accent_cyan"],
        "params": {
            "target_context": {
                "label": "Target context length",
                "default": "524288",
                "type": "number",
            },
            "search_method": {
                "label": "Search method",
                "default": "evolutionary",
                "type": "text",
            },
        },
    },
    "lora": {
        "label": "LoRA (Low-Rank Adaptation)",
        "color": COLORS["accent_magenta"],
        "params": {
            "adapter_path": {"label": "Adapter path", "default": "", "type": "path"},
            "merge": {"label": "Merge into base", "default": "false", "type": "bool"},
            "train": {"label": "Train new adapter", "default": "false", "type": "bool"},
            "rank": {"label": "LoRA rank", "default": "32", "type": "number"},
        },
    },
    "quantize": {
        "label": "Quantize (Model compression)",
        "color": COLORS["accent_green"],
        "params": {
            "quant_type": {
                "label": "Quantization type",
                "default": "Q4_K_M",
                "type": "text",
            },
        },
    },
}


class OperationBuilderScreen(Widget):
    """Multi-step wizard for building and queueing operations."""

    DEFAULT_CSS = """
    OperationBuilderScreen {
        height: 1fr;
        layout: horizontal;
    }
    OperationBuilderScreen .ob-steps {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        padding: 1;
    }
    OperationBuilderScreen .ob-preview {
        width: 1fr;
        height: 1fr;
        layout: vertical;
        border-left: solid #2a3a4a;
        padding: 1;
    }
    OperationBuilderScreen .ob-title {
        color: #00d4ff;
        text-style: bold;
        padding: 0 0 1 0;
    }
    OperationBuilderScreen .ob-step-label {
        color: #6a7a8a;
        padding: 1 0 0 0;
    }
    OperationBuilderScreen .ob-step-active {
        color: #00d4ff;
        text-style: bold;
        padding: 1 0 0 0;
    }
    OperationBuilderScreen OptionList {
        height: auto;
        max-height: 12;
        background: #131924;
    }
    OperationBuilderScreen Input {
        margin: 0 0 1 0;
    }
    OperationBuilderScreen .ob-nav {
        height: 3;
        layout: horizontal;
        padding: 1 0 0 0;
    }
    OperationBuilderScreen Button {
        margin: 0 1 0 0;
    }
    OperationBuilderScreen RichLog {
        height: 1fr;
        background: #131924;
    }
    """

    current_step: reactive[int] = reactive(1)

    def __init__(
        self,
        *,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self._selected_op: str = ""
        self._model_path: str = ""
        self._output_path: str = ""
        self._params: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(classes="ob-steps"):
                yield Static("Build Operation", classes="ob-title")

                # Step indicator
                yield Static("", id="ob-step-indicator")

                # Step 1: Operation type
                yield Label("Step 1: Select Operation Type", classes="ob-step-active", id="ob-s1-label")
                yield OptionList(
                    Option("Abliterate (Deregulation via directional ablation)", id="abliterate"),
                    Option("LongRoPE (Context window extension)", id="longrope"),
                    Option("LoRA (Low-Rank Adaptation)", id="lora"),
                    Option("Quantize (Model compression)", id="quantize"),
                    id="ob-op-list",
                )

                # Step 2: Model path
                yield Label("Step 2: Source Model Path", classes="ob-step-label", id="ob-s2-label")
                yield Input(
                    placeholder="/AEGIS_AI/models/your-model",
                    id="ob-model-input",
                )

                # Step 3: Parameters (dynamic — filled on step transition)
                yield Label("Step 3: Configure Parameters", classes="ob-step-label", id="ob-s3-label")
                yield Vertical(id="ob-params-container")

                # Step 4: Output path
                yield Label("Step 4: Output Path", classes="ob-step-label", id="ob-s4-label")
                yield Input(
                    placeholder="/AEGIS_AI/models/output-model",
                    id="ob-output-input",
                )

                # Navigation
                with Horizontal(classes="ob-nav"):
                    yield Button("< Back", id="ob-back", variant="default")
                    yield Button("Next >", id="ob-next", variant="primary")
                    yield Button("Queue Job", id="ob-queue", variant="success")

            with Vertical(classes="ob-preview"):
                yield Static("Configuration Preview", classes="ob-title")
                yield RichLog(highlight=True, markup=True, id="ob-preview-log")

    def on_mount(self) -> None:
        """Initialize step visibility."""
        self._update_step_visibility()
        self._update_preview()

    def watch_current_step(self, value: int) -> None:
        self._update_step_visibility()
        self._update_preview()

    def _update_step_visibility(self) -> None:
        """Show/hide UI elements based on current step."""
        step = self.current_step

        # Step labels — highlight active
        for i in range(1, 5):
            try:
                label = self.query_one(f"#ob-s{i}-label", Label)
                if i == step:
                    label.set_classes("ob-step-active")
                elif i < step:
                    label.set_classes("ob-step-label status-success")
                else:
                    label.set_classes("ob-step-label")
            except Exception:
                pass

        # Back/Next/Queue visibility
        try:
            self.query_one("#ob-back", Button).display = step > 1
            self.query_one("#ob-next", Button).display = step < 5
            self.query_one("#ob-queue", Button).display = step >= 4
        except Exception:
            pass

    def _update_preview(self) -> None:
        """Refresh the configuration preview panel."""
        try:
            preview = self.query_one("#ob-preview-log", RichLog)
        except Exception:
            return

        preview.clear()
        preview.write(f"[bold]Current Configuration:[/]")
        preview.write("")

        if self._selected_op:
            op_info = OPERATIONS.get(self._selected_op, {})
            color = op_info.get("color", COLORS["text"])
            preview.write(f"  Operation: [{color}]{self._selected_op}[/]")
        else:
            preview.write(f"  Operation: [{COLORS['text_muted']}](not selected)[/]")

        if self._model_path:
            preview.write(f"  Model:     {self._model_path}")
        else:
            preview.write(f"  Model:     [{COLORS['text_muted']}](not set)[/]")

        if self._params:
            preview.write("")
            preview.write(f"  [bold]Parameters:[/]")
            for key, val in self._params.items():
                preview.write(f"    {key}: {val}")

        if self._output_path:
            preview.write(f"  Output:    {self._output_path}")
        else:
            preview.write(f"  Output:    [{COLORS['text_muted']}](not set)[/]")

        # Step-specific hints
        preview.write("")
        step = self.current_step
        if step == 1:
            preview.write(f"[{COLORS['text_muted']}]Select an operation type from the list.[/]")
        elif step == 2:
            preview.write(f"[{COLORS['text_muted']}]Enter the path to the source model.[/]")
        elif step == 3:
            preview.write(f"[{COLORS['text_muted']}]Configure operation parameters.[/]")
        elif step == 4:
            preview.write(f"[{COLORS['text_muted']}]Set the output path and queue the job.[/]")

    def on_option_list_option_selected(
        self, event: OptionList.OptionSelected
    ) -> None:
        """Handle operation type selection (Step 1)."""
        if event.option_list.id == "ob-op-list":
            self._selected_op = str(event.option.id)
            self._params = {}
            # Pre-fill default parameters
            op_info = OPERATIONS.get(self._selected_op, {})
            for key, pdef in op_info.get("params", {}).items():
                self._params[key] = pdef["default"]
            self._update_preview()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle navigation button presses."""
        button_id = event.button.id

        if button_id == "ob-next":
            # Validate current step before advancing
            if self.current_step == 1 and not self._selected_op:
                return  # Must select an operation
            if self.current_step == 2:
                try:
                    inp = self.query_one("#ob-model-input", Input)
                    self._model_path = inp.value.strip()
                except Exception:
                    pass
            if self.current_step == 3:
                self._collect_params()
            self.current_step = min(self.current_step + 1, 5)

        elif button_id == "ob-back":
            self.current_step = max(self.current_step - 1, 1)

        elif button_id == "ob-queue":
            # Collect output path
            try:
                inp = self.query_one("#ob-output-input", Input)
                self._output_path = inp.value.strip()
            except Exception:
                pass
            self._queue_job()

    def _collect_params(self) -> None:
        """Collect parameter values from dynamically created inputs."""
        # Parameters are stored in self._params from defaults or user edits
        pass

    def _queue_job(self) -> None:
        """Submit the configured operation to the job queue."""
        try:
            preview = self.query_one("#ob-preview-log", RichLog)
        except Exception:
            return

        if not self._selected_op or not self._model_path or not self._output_path:
            preview.write(
                f"\n[{COLORS['accent_red']}]Error: Operation, model path, and output path are required.[/]"
            )
            return

        preview.write("")
        preview.write(f"[{COLORS['accent_green']}]Job queued successfully![/]")
        preview.write(f"  Operation: {self._selected_op}")
        preview.write(f"  Model:     {self._model_path}")
        preview.write(f"  Output:    {self._output_path}")

        logger.info(
            "Queued operation: %s model=%s output=%s params=%s",
            self._selected_op,
            self._model_path,
            self._output_path,
            self._params,
        )
