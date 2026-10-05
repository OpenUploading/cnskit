"""Sparse graph binding. Matrix rows receive from columns; no dense NxN copy."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass
class Graph:
    body_ids: np.ndarray
    weights: sparse.csr_matrix
    provenance: dict

    def __post_init__(self):
        ids = np.asarray(self.body_ids)
        if (
            ids.ndim != 1
            or not len(ids)
            or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("body_ids must be unique nonempty integers")
        if ids.dtype.kind == "u" and np.any(ids > np.iinfo(np.int64).max):
            raise ValueError("body_ids must fit signed 64-bit integers")
        self.body_ids = ids.astype(np.int64, copy=True)
        self.weights = sparse.csr_matrix(self.weights, dtype=np.float32, copy=True)
        self.weights.sum_duplicates()
        self.weights.sort_indices()
        self.weights.eliminate_zeros()
        if self.weights.shape != (len(ids), len(ids)) or not np.isfinite(self.weights.data).all():
            raise ValueError("Graph dimensions or weights invalid")
        self.provenance = json.loads(json.dumps(self.provenance, allow_nan=False))

    @property
    def fingerprint(self):
        h = hashlib.sha256()
        for a in (
            self.body_ids.astype("<i8"),
            self.weights.indptr.astype("<i8"),
            self.weights.indices.astype("<i8"),
            self.weights.data.astype("<f4"),
        ):
            h.update(a.tobytes())
        return h.hexdigest()

    def indices(self, ids):
        ids = list(ids)
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(type(x) not in (int, np.int64, np.int32) for x in ids)
        ):
            raise ValueError("Select nonempty unique integer body IDs")
        lookup = {int(x): i for i, x in enumerate(self.body_ids)}
        try:
            return np.array([lookup[x] for x in ids], dtype=np.int64)
        except KeyError as e:
            raise ValueError(f"Unknown body ID: {e.args[0]}") from e

    @classmethod
    def from_malecns(cls, directory, *, body_ids=None):
        from cns_tinker.runtime.malecns import MaleCNSSession

        with MaleCNSSession(directory) as session:
            graph = cls(session.body_ids, session._weights, session.truth_metadata())
        return graph if body_ids is None else graph.subgraph(body_ids)

    def subgraph(self, body_ids):
        idx = self.indices(body_ids)
        return Graph(
            self.body_ids[idx],
            self.weights[idx][:, idx],
            {
                **self.provenance,
                "parent_fingerprint": self.fingerprint,
                "selection": (
                    "induced subgraph; parent normalization retained; omitted edges removed"
                ),
            },
        )

    @classmethod
    def synthetic(cls, n=64, density=0.08, seed=0):
        if type(n) is not int or n < 2 or not 0 < density <= 1:
            raise ValueError("Invalid synthetic graph size or density")
        rng = np.random.default_rng(seed)
        w = sparse.random(n, n, density=density, random_state=rng, format="csr", dtype=np.float32)
        sums = np.asarray(w.sum(axis=1)).ravel()
        w.data /= np.repeat(np.maximum(sums, 1), np.diff(w.indptr))
        return cls(
            np.arange(n, dtype=np.int64),
            w,
            {
                "dataset": "synthetic",
                "seed": seed,
                "biological_validation": False,
                "matrix_orientation": "row=post,column=pre",
            },
        )
