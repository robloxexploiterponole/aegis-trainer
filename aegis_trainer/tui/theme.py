"""
AEGIS color theme constants and Textual CSS.

Provides the unified visual identity for both the Textual TUI and Rich CLI
output. Colors are calibrated for dark terminals with true-color support.

Layer type color coding:
  - DeltaNet (linear_attention): accent_yellow (#ffd700)
  - Full Attention (RoPE/GQA):   accent_cyan (#00d4ff)

Resource bar thresholds:
  - 0-60%:  green  (#00ff88)
  - 60-80%: yellow (#ffd700)
  - 80-90%: orange (#ff8800)
  - 90-100%: red   (#ff4444)

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

from aegis_trainer import __version__

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

COLORS = {
    # Backgrounds
    "background": "#0a0e14",
    "surface": "#131924",
    "surface_raised": "#1a2233",
    "border": "#2a3a4a",

    # Text
    "text": "#d4dae4",
    "text_muted": "#6a7a8a",

    # Accents
    "accent_cyan": "#00d4ff",
    "accent_green": "#00ff88",
    "accent_yellow": "#ffd700",
    "accent_red": "#ff4444",
    "accent_magenta": "#cc66ff",

    # Resource bar thresholds
    "resource_green": "#00ff88",
    "resource_yellow": "#ffd700",
    "resource_orange": "#ff8800",
    "resource_red": "#ff4444",
}

# Convenience aliases for layer-type coloring
LAYER_COLORS = {
    "linear_attention": COLORS["accent_yellow"],   # DeltaNet
    "full_attention": COLORS["accent_cyan"],        # Standard attention
}

# ---------------------------------------------------------------------------
# ASCII art header
# ---------------------------------------------------------------------------

HEADER_ART = rf"""    ░█████╗░███████╗░██████╗░██╗░██████╗
    ██╔══██╗██╔════╝██╔════╝░██║██╔════╝
    ███████║█████╗░░██║░░██╗░██║╚█████╗░
    ██╔══██║██╔══╝░░██║░░╚██╗██║░╚═══██╗
    ██║░░██║███████╗╚██████╔╝██║██████╔╝
    ╚═╝░░╚═╝╚══════╝░╚═════╝░╚═╝╚═════╝░  AI Trainer v{__version__}
    Hardwick Software Services"""


def header_rich() -> str:
    """Return the ASCII header wrapped in Rich markup (cyan)."""
    return f"[bold cyan]{HEADER_ART}[/bold cyan]"


# ---------------------------------------------------------------------------
# Resource bar color helper
# ---------------------------------------------------------------------------

def resource_color(percent: float) -> str:
    """Return the theme color string for a given resource-usage percentage.

    Args:
        percent: Usage percentage (0-100).

    Returns:
        Hex color string from the resource bar palette.
    """
    if percent >= 90.0:
        return COLORS["resource_red"]
    if percent >= 80.0:
        return COLORS["resource_orange"]
    if percent >= 60.0:
        return COLORS["resource_yellow"]
    return COLORS["resource_green"]


# ---------------------------------------------------------------------------
# Textual CSS
# ---------------------------------------------------------------------------

AEGIS_CSS = """
Screen {
    background: #0a0e14;
}

Header {
    background: #131924;
    color: #00d4ff;
}

Footer {
    background: #131924;
    color: #6a7a8a;
}

TabbedContent {
    background: #0a0e14;
}

TabPane {
    background: #0a0e14;
    padding: 1;
}

ContentSwitcher {
    background: #0a0e14;
}

Tabs {
    background: #131924;
}

Tab {
    background: #131924;
    color: #6a7a8a;
}

Tab.-active {
    background: #1a2233;
    color: #00d4ff;
}

Tab:hover {
    background: #1a2233;
    color: #d4dae4;
}

DataTable {
    background: #131924;
    color: #d4dae4;
}

DataTable > .datatable--header {
    background: #1a2233;
    color: #00d4ff;
}

DataTable > .datatable--cursor {
    background: #2a3a4a;
    color: #d4dae4;
}

RichLog {
    background: #131924;
    color: #d4dae4;
    border: solid #2a3a4a;
    padding: 0 1;
}

Input {
    background: #1a2233;
    color: #d4dae4;
    border: solid #2a3a4a;
}

Input:focus {
    border: solid #00d4ff;
}

Button {
    background: #1a2233;
    color: #d4dae4;
    border: solid #2a3a4a;
}

Button:hover {
    background: #2a3a4a;
    color: #00d4ff;
}

Button.-primary {
    background: #00d4ff;
    color: #0a0e14;
}

Select {
    background: #1a2233;
    color: #d4dae4;
    border: solid #2a3a4a;
}

OptionList {
    background: #1a2233;
    color: #d4dae4;
}

OptionList > .option-list--option-highlighted {
    background: #2a3a4a;
    color: #00d4ff;
}

Static {
    background: transparent;
    color: #d4dae4;
}

ProgressBar {
    background: #131924;
}

ProgressBar Bar {
    color: #00d4ff;
    background: #2a3a4a;
}

/* Widget-specific classes */
.resource-bar-container {
    height: 3;
    padding: 0 1;
}

.header-art {
    color: #00d4ff;
    text-align: center;
    padding: 1 0;
}

.panel-title {
    color: #00d4ff;
    text-style: bold;
    padding: 0 1;
}

.layer-deltanet {
    color: #ffd700;
}

.layer-attention {
    color: #00d4ff;
}

.layer-completed {
    color: #00ff88;
}

.layer-error {
    color: #ff4444;
}

.op-lora {
    color: #cc66ff;
}

.status-success {
    color: #00ff88;
}

.status-warning {
    color: #ffd700;
}

.status-error {
    color: #ff4444;
}

.muted {
    color: #6a7a8a;
}
"""

# ---------------------------------------------------------------------------
# Rich theme dict (for Console(theme=...) if desired)
# ---------------------------------------------------------------------------

RICH_THEME = {
    "aegis.bg": f"on {COLORS['background']}",
    "aegis.surface": f"on {COLORS['surface']}",
    "aegis.text": COLORS["text"],
    "aegis.muted": COLORS["text_muted"],
    "aegis.cyan": COLORS["accent_cyan"],
    "aegis.green": COLORS["accent_green"],
    "aegis.yellow": COLORS["accent_yellow"],
    "aegis.red": COLORS["accent_red"],
    "aegis.magenta": COLORS["accent_magenta"],
    "aegis.deltanet": f"bold {COLORS['accent_yellow']}",
    "aegis.attention": f"bold {COLORS['accent_cyan']}",
}
