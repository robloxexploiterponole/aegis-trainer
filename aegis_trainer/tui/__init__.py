"""
Terminal User Interface for AEGIS AI Trainer.

Exports the main AegisTrainerApp Textual application class.
Import is lazy to avoid requiring ``textual`` when only the CLI
or non-TUI components are used.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

__all__ = ["AegisTrainerApp"]


def __getattr__(name: str):
    if name == "AegisTrainerApp":
        from aegis_trainer.tui.app import AegisTrainerApp
        return AegisTrainerApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
