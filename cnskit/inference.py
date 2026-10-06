"""File-backed prediction without loading whole sequences or outputs into RAM."""

import os
import tempfile
from pathlib import Path

import numpy as np


def predict_file(model, input_path, output_path, *, chunk_size=1024):
    """Stream a numeric NPY [episode,time,input] into a new prediction NPY.

    Each episode starts with fresh state. The destination appears only after
    successful inference; existing files are never replaced. Publication uses
    a same-directory hard link and requires filesystem support for hard links.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    x = np.load(input_path, mmap_mode="r", allow_pickle=False)
    temporary = None
    try:
        if not isinstance(x, np.memmap):
            raise ValueError("Input must be an uncompressed numeric NPY file")
        if (
            x.ndim != 3
            or min(x.shape) < 1
            or x.shape[2] != model.config["input_size"]
            or x.dtype.kind not in "biuf"
        ):
            raise ValueError("Expected real numeric [episode,time,input_size] observations")
        shape = (x.shape[0], x.shape[1], model.config["output_size"])
        with tempfile.NamedTemporaryFile(
            prefix=".cnskit-predict-", suffix=".tmp", dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.lib.format.write_array_header_2_0(
                stream, {"descr": "<f4", "fortran_order": False, "shape": shape}
            )
            for episode in range(len(x)):
                state = None
                for left in range(0, x.shape[1], chunk_size):
                    chunk = np.array(
                        x[episode : episode + 1, left : left + chunk_size],
                        dtype=np.float32,
                        copy=True,
                    )
                    prediction, state = model.predict(chunk, state=state)
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError("Nonfinite model predictions")
                    stream.write(prediction.astype("<f4", copy=False).tobytes(order="C"))
            stream.flush()
            os.fsync(stream.fileno())
        # Atomic, no-clobber publication even if another writer created the path.
        os.link(temporary, destination)
        return {
            "episodes": shape[0],
            "steps_per_episode": shape[1],
            "outputs": shape[2],
            "chunk_size": chunk_size,
        }
    finally:
        if isinstance(x, np.memmap):
            x._mmap.close()
        elif hasattr(x, "close"):
            x.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
