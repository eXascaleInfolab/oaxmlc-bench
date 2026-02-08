from pathlib import Path
import torch
from typing import Callable, Dict, Any, Optional, Tuple


class CheckpointManager:
    def __init__(self, root: Path, keep_last: int = 1, prefix: str = "epoch"):
        """Create a manager that saves checkpoints under `root`.

        Args:
            root: Directory that will hold checkpoint files.
            keep_last: Maximum number of checkpoints to keep; older ones are pruned.
            prefix: Filename prefix (e.g. `epoch_0001.pt`).
        """
        self.root = root
        self.keep_last = keep_last
        self.prefix = prefix
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, *, epoch: int, model_saver: Callable[[Path], None], state: Dict[str, Any]) -> Path:
        """Persist the model weights and extra run state for a given epoch.

        `model_saver` should handle writing the actual model parameters to the returned path.
        We also store a sidecar `.meta` file with the epoch index and any additional state
        (optimizer state, metrics, etc.).

        Args:
            epoch: Epoch number used in the filename.
            model_saver: Callable that writes weights to the target path.
            state: Extra metadata to serialize alongside the epoch number.

        Returns:
            Path to the weight file that was written.
        """
        file = self.root / f"{self.prefix}_{epoch:04d}.pt"
        model_saver(file)
        torch.save({"epoch": epoch, **state}, file.with_suffix(".meta"))
        self._prune()
        return file

    def latest(self):
        """Load the most recent checkpoint metadata and return paths + stored state.

        Returns:
            Dict with the epoch number, resolved path to the weight file, and the
            loaded metadata dictionary; `None` if no checkpoints exist.
        """
        metas = sorted(self.root.glob("*.meta"))
        if not metas:
            return None
        latest_meta = metas[-1]
        state = torch.load(latest_meta)
        return {
            "epoch": state["epoch"],
            "model_path": latest_meta.with_suffix(".pt"),
            "state": state,
        }

    def _prune(self) -> None:
        """Remove older checkpoints beyond the `keep_last` retention window."""
        checkpoints = sorted(self.root.glob(f"{self.prefix}_*.pt"))
        for old_ckpt in checkpoints[:-self.keep_last]:
            old_ckpt.unlink(missing_ok=True)
            meta = old_ckpt.with_suffix(".meta")
            meta.unlink(missing_ok=True)