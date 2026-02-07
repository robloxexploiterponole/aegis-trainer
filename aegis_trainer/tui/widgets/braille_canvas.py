"""
BrailleCanvas — Unicode braille dot-matrix rendering engine.

Renders 2D scatter plots using Unicode braille characters (U+2800-U+28FF).
Each character cell contains a 2x4 dot grid, providing 2x horizontal and
4x vertical sub-character resolution for smooth scatter plots in the terminal.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations


# Unicode braille base codepoint (empty braille pattern)
BRAILLE_BASE = 0x2800

# Bit masks for each dot position within a 2x4 braille cell.
# Layout:
#   (col=0, row=0) -> 0x01    (col=1, row=0) -> 0x08
#   (col=0, row=1) -> 0x02    (col=1, row=1) -> 0x10
#   (col=0, row=2) -> 0x04    (col=1, row=2) -> 0x20
#   (col=0, row=3) -> 0x40    (col=1, row=3) -> 0x80
_DOT_MAP: list[list[int]] = [
    # col 0          col 1
    [0x01, 0x08],  # row 0
    [0x02, 0x10],  # row 1
    [0x04, 0x20],  # row 2
    [0x40, 0x80],  # row 3
]


class BrailleCanvas:
    """Low-level braille dot-matrix renderer.

    Each character cell maps to a 2-wide x 4-tall grid of dots. The canvas
    stores dot presence per pixel and one color per character cell (the most
    recently written dot's color wins).

    Args:
        width_chars: Canvas width measured in terminal character columns.
        height_chars: Canvas height measured in terminal character rows.
    """

    def __init__(self, width_chars: int, height_chars: int) -> None:
        self.width_chars = max(1, width_chars)
        self.height_chars = max(1, height_chars)

        # Pixel dimensions (sub-character resolution)
        self.pixel_width = self.width_chars * 2
        self.pixel_height = self.height_chars * 4

        # Dot grid: True means dot is set
        self._dots: list[list[bool]] = [
            [False] * self.pixel_width for _ in range(self.pixel_height)
        ]

        # One color per character cell (most recently set dot wins)
        self._colors: list[list[str]] = [
            [""] * self.width_chars for _ in range(self.height_chars)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Reset all dots and colors to empty."""
        for row in self._dots:
            for i in range(len(row)):
                row[i] = False
        for row in self._colors:
            for i in range(len(row)):
                row[i] = ""

    def set_dot(self, x: int, y: int, color: str = "") -> None:
        """Set a single dot at pixel coordinates.

        Args:
            x: Horizontal pixel position (0 = left).
            y: Vertical pixel position (0 = top).
            color: Optional Rich-compatible color string (e.g. ``"#00d4ff"``).
        """
        if 0 <= x < self.pixel_width and 0 <= y < self.pixel_height:
            self._dots[y][x] = True
            if color:
                # Map pixel to character cell
                cx = x // 2
                cy = y // 4
                self._colors[cy][cx] = color

    def set_point(
        self,
        x: float,
        y: float,
        color: str = "",
        x_range: tuple[float, float] = (0.0, 1.0),
        y_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Map a normalized / ranged coordinate to a dot position and set it.

        This is the main entry point for scatter-plot use: pass real-valued
        coordinates plus their data ranges, and the method maps them onto
        the pixel grid.

        Args:
            x: Horizontal value in ``x_range``.
            y: Vertical value in ``y_range``.
            color: Optional Rich-compatible color string.
            x_range: ``(min, max)`` of the horizontal axis.
            y_range: ``(min, max)`` of the vertical axis.
        """
        x_min, x_max = x_range
        y_min, y_max = y_range

        # Avoid division by zero
        x_span = x_max - x_min if x_max != x_min else 1.0
        y_span = y_max - y_min if y_max != y_min else 1.0

        # Normalize to [0, 1]
        nx = (x - x_min) / x_span
        ny = (y - y_min) / y_span

        # Clamp
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))

        # Convert to pixel indices (y is inverted: 0 at top in terminal)
        px = int(nx * (self.pixel_width - 1))
        py = int((1.0 - ny) * (self.pixel_height - 1))

        self.set_dot(px, py, color)

    def render(self) -> str:
        """Produce a Rich markup string with colored braille characters.

        Each character row becomes one line of output. Colors are applied
        per character cell using Rich ``[color]...[/]`` tags.  Empty cells
        (no dots set) are rendered as the braille space character (U+2800).

        Returns:
            Multi-line string with Rich markup ready for display.
        """
        lines: list[str] = []

        for cy in range(self.height_chars):
            parts: list[str] = []
            for cx in range(self.width_chars):
                code = BRAILLE_BASE
                # Accumulate bits for this cell
                for row in range(4):
                    for col in range(2):
                        px = cx * 2 + col
                        py = cy * 4 + row
                        if self._dots[py][px]:
                            code |= _DOT_MAP[row][col]

                char = chr(code)
                cell_color = self._colors[cy][cx]
                if cell_color and code != BRAILLE_BASE:
                    parts.append(f"[{cell_color}]{char}[/]")
                else:
                    parts.append(char)
            lines.append("".join(parts))

        return "\n".join(lines)

    def render_with_axes(
        self,
        title: str = "",
        x_label: str = "",
        y_label: str = "",
    ) -> str:
        """Render the braille canvas with optional axis labels and title.

        The title is centered above the plot, the y_label is placed on the
        left side, and the x_label is centered below the plot.

        Args:
            title: Optional title displayed above the plot.
            x_label: Optional label below the x-axis.
            y_label: Optional label to the left of the y-axis.

        Returns:
            Multi-line Rich markup string.
        """
        body = self.render()
        body_lines = body.split("\n")
        output_lines: list[str] = []

        # Title
        if title:
            pad = max(0, (self.width_chars - len(title)) // 2)
            output_lines.append(f"{'':>{pad}}{title}")
            output_lines.append("")

        # Y-axis label (placed to the left of the middle row)
        y_chars = list(y_label) if y_label else []
        y_offset = max(0, (self.height_chars - len(y_chars)) // 2)

        for i, line in enumerate(body_lines):
            prefix = ""
            y_idx = i - y_offset
            if 0 <= y_idx < len(y_chars):
                prefix = y_chars[y_idx] + " "
            elif y_label:
                prefix = "  "
            output_lines.append(f"{prefix}{line}")

        # X-axis label
        if x_label:
            pad = max(0, (self.width_chars - len(x_label)) // 2)
            prefix_width = 2 if y_label else 0
            output_lines.append("")
            output_lines.append(f"{'':>{prefix_width + pad}}{x_label}")

        return "\n".join(output_lines)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def dot_count(self) -> int:
        """Return the total number of dots currently set."""
        return sum(1 for row in self._dots for d in row if d)

    def __repr__(self) -> str:
        return (
            f"BrailleCanvas({self.width_chars}x{self.height_chars} chars, "
            f"{self.pixel_width}x{self.pixel_height} dots, "
            f"{self.dot_count()} set)"
        )
