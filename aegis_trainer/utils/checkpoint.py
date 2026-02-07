"""
CheckpointManager — Resumable run tracking for AEGIS AI Trainer.

Persists a JSON file at {output_path}/.aegis_checkpoint.json that records
which layers have been processed. Enables crash-safe resumption — if the
trainer is interrupted, it can skip already-completed layers on restart.

Author: Hardwick Software Services (justcalljon.pro)
GitHub: github.com/jonhardwick-spec/aegis-trainer
License: SSPL-1.0
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_CHECKPOINT_FILENAME = ".aegis_checkpoint.json"


class CheckpointManager:
    """Track and persist layer completion state for resumable training runs.

    Stores a JSON checkpoint at {output_path}/.aegis_checkpoint.json with:
      - completed_layers: list of layer names that have been fully processed
      - metadata: optional run metadata (operations, start time, etc.)

    Usage::

        ckpt = CheckpointManager(output_path=Path("/output/splitted_model"))

        for layer_name in layer_names:
            if ckpt.is_completed(layer_name):
                print(f"Skipping {layer_name} (already done)")
                continue
            # ... process layer ...
            ckpt.mark_completed(layer_name)
    """

    def __init__(
        self,
        output_path: Path | str,
        run_id: Optional[str] = None,
    ):
        """Initialize the checkpoint manager.

        Args:
            output_path: Directory where the checkpoint file is stored.
            run_id: Optional identifier for this run. If provided, the
                checkpoint file will only be used if the run_id matches.
        """
        self.output_path = Path(output_path)
        self.checkpoint_file = self.output_path / _CHECKPOINT_FILENAME
        self.run_id = run_id

        self._completed: Set[str] = set()
        self._metadata: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load checkpoint state from disk if it exists."""
        if not self.checkpoint_file.exists():
            logger.debug("No checkpoint file found at %s", self.checkpoint_file)
            return

        try:
            data = json.loads(self.checkpoint_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Failed to read checkpoint file %s: %s. Starting fresh.",
                self.checkpoint_file,
                exc,
            )
            return

        # If run_id is specified, only restore if it matches
        stored_run_id = data.get("run_id")
        if self.run_id is not None and stored_run_id != self.run_id:
            logger.info(
                "Checkpoint run_id mismatch (stored=%s, current=%s). Starting fresh.",
                stored_run_id,
                self.run_id,
            )
            return

        self._completed = set(data.get("completed_layers", []))
        self._metadata = data.get("metadata", {})
        logger.info(
            "Restored checkpoint: %d layers completed", len(self._completed)
        )

    def _save(self) -> None:
        """Persist current checkpoint state to disk.

        Creates the output directory if needed. Writes atomically by
        writing to a temp file then renaming.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        data = {
            "run_id": self.run_id,
            "completed_layers": sorted(self._completed),
            "metadata": self._metadata,
            "last_updated": time.time(),
            "num_completed": len(self._completed),
        }

        # Write to temp file first for atomic save
        tmp_file = self.checkpoint_file.with_suffix(".json.tmp")
        try:
            tmp_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_file.replace(self.checkpoint_file)
        except OSError as exc:
            logger.error("Failed to save checkpoint: %s", exc)
            # Clean up temp file on failure
            if tmp_file.exists():
                tmp_file.unlink()
            raise

    def is_completed(self, layer_name: str) -> bool:
        """Check if a layer has been marked as completed.

        Args:
            layer_name: Layer name string (e.g. "model.layers.0.").

        Returns:
            True if the layer has been processed in this run.
        """
        return layer_name in self._completed

    def mark_completed(self, layer_name: str) -> None:
        """Mark a layer as completed and persist to disk.

        Args:
            layer_name: Layer name string to mark as done.
        """
        self._completed.add(layer_name)
        self._save()
        logger.debug("Marked layer %s as completed (%d total)", layer_name, len(self._completed))

    def get_completed(self) -> List[str]:
        """Get list of all completed layer names, sorted.

        Returns:
            Sorted list of completed layer name strings.
        """
        return sorted(self._completed)

    @property
    def num_completed(self) -> int:
        """Number of layers completed so far."""
        return len(self._completed)

    def set_metadata(self, key: str, value: Any) -> None:
        """Store arbitrary metadata in the checkpoint.

        Useful for recording run configuration, operation names, etc.

        Args:
            key: Metadata key.
            value: JSON-serializable value.
        """
        self._metadata[key] = value
        self._save()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Retrieve metadata from the checkpoint.

        Args:
            key: Metadata key.
            default: Default value if key not found.

        Returns:
            The stored value, or default.
        """
        return self._metadata.get(key, default)

    def reset(self) -> None:
        """Delete the checkpoint file and clear all state.

        Use this to force a full re-run.
        """
        self._completed.clear()
        self._metadata.clear()

        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Deleted checkpoint file: %s", self.checkpoint_file)
        else:
            logger.debug("No checkpoint file to delete at %s", self.checkpoint_file)
