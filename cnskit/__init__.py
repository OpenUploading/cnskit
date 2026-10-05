"""Task training and inference on explicit connectome graphs."""

from typing import TYPE_CHECKING

from .graph import Graph

if TYPE_CHECKING:
    from .learning import ConnectomeModel, Episode, Trainer, load_policy

__all__ = ["Graph", "ConnectomeModel", "Episode", "Trainer", "load_policy"]


def __getattr__(name):
    if name in {"ConnectomeModel", "Trainer", "Episode", "load_policy"}:
        from . import learning

        return getattr(learning, name)
    raise AttributeError(name)
