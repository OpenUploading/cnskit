"""Task training and inference on explicit connectome graphs."""

from .graph import Graph

__all__ = ["Graph"]


def __getattr__(name):
    if name in {"ConnectomeModel", "Trainer", "Episode", "load_policy"}:
        from . import learning

        return getattr(learning, name)
    raise AttributeError(name)
