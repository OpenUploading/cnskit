"""Official MaleCNS v1.0 graph ingestion and explicitly synthetic dynamics.

No private runtime code or datasets are used. Raw weights remain synapse counts.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from cns_tinker.runtime.connectome import RuntimeState

ANNOTATIONS = 'body-annotations-male-cns-v1.0-minconf-0.5.feather'
WEIGHTS = 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
SOURCE = 'https://male-cns.janelia.org/download/'
BASE = 'https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(data_dir: str | Path, output: str | Path) -> dict:
    """Stream the official table; keep only edges between status=Traced bodies."""
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    from scipy import sparse

    data_dir, output = Path(data_dir), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use an empty output directory; existing graphs are not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    table = feather.read_table(data_dir / ANNOTATIONS).to_pandas()
    nodes = table.loc[table.status.eq('Traced')].sort_values('bodyId')
    ids = nodes.bodyId.to_numpy(dtype=np.int64)
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError('Empty or duplicate traced body IDs')
    rows, cols, values = [], [], []
    source_rows = 0
    with pa.memory_map(str(data_dir / WEIGHTS)) as source:
        reader = ipc.open_file(source)
        if reader.schema.names != ['body_pre', 'body_post', 'weight']:
            raise ValueError('Unexpected connectivity schema')
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            pre, post, weight = [batch.column(j).to_numpy() for j in range(3)]
            source_rows += len(weight)
            if np.any(weight <= 0):
                raise ValueError('Nonpositive source synapse count')
            a, b = np.searchsorted(ids, pre), np.searchsorted(ids, post)
            valid = (a < len(ids)) & (b < len(ids))
            keep = np.flatnonzero(valid)
            keep = keep[(ids[a[keep]] == pre[keep]) & (ids[b[keep]] == post[keep])]
            # Matrix rows receive from columns: W[post, pre].
            rows.append(b[keep].astype(np.int32))
            cols.append(a[keep].astype(np.int32))
            values.append(weight[keep].astype(np.int64))
    graph = sparse.coo_matrix((np.concatenate(values),
                              (np.concatenate(rows), np.concatenate(cols))),
                             shape=(len(ids), len(ids)), dtype=np.int64).tocsr()
    graph.sum_duplicates()
    sparse.save_npz(output / 'synapse_counts.npz', graph)
    np.save(output / 'body_ids.npy', ids, allow_pickle=False)
    nodes[['bodyId', 'type', 'instance', 'superclass', 'status']].to_csv(output / 'nodes.csv', index=False)
    manifest = {
        'dataset': 'male-cns:v1.0', 'source_page': SOURCE,
        'license': 'CC-BY-4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/',
        'attribution': 'FlyEM HHMI Janelia, University of Cambridge, MRC LMB, Google Research',
        'selection': 'status == Traced; induced graph; all other segments excluded',
        'node_count': len(ids), 'edge_count': int(graph.nnz),
        'synapse_count': int(graph.sum()), 'source_connection_rows': source_rows,
        'matrix_orientation': 'row=body_post, column=body_pre',
        'weight_units': 'synapse count', 'biological_validation': False,
        'sources': {name: {'url': BASE + name, 'sha256': sha256(data_dir / name)}
                    for name in (ANNOTATIONS, WEIGHTS)},
        'artifacts': {name: sha256(output / name)
                      for name in ('synapse_counts.npz', 'body_ids.npy', 'nodes.csv')},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class MaleCNSSession:
    """Real traced-body graph with an unsigned, normalized leaky-rate model.

    Currents are injected into explicit body IDs in arbitrary units. This is not
    a calibrated sensory model. No inferred neurotransmitter signs are applied.
    No automatic game controller or trained readout is attached.
    """

    def __init__(self, graph_dir: str | Path, *, dt_ms: float = 5.0,
                 gain: float = 0.8, workers: int = 1):
        from scipy import sparse

        if not math.isfinite(dt_ms) or not 0 < dt_ms <= 20:
            raise ValueError('dt_ms must be finite and in (0, 20]')
        if not math.isfinite(gain) or not 0 <= gain < 1:
            raise ValueError('gain must be finite and in [0, 1)')
        if type(workers) is not int or not 1 <= workers <= 32:
            raise ValueError('workers must be an integer in [1, 32]')
        root = Path(graph_dir)
        self._graph_dir = root
        self.manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        if self.manifest.get('dataset') != 'male-cns:v1.0':
            raise ValueError('Expected MaleCNS v1.0 provenance')
        for name in ('synapse_counts.npz', 'body_ids.npy', 'nodes.csv'):
            if sha256(root / name) != self.manifest['artifacts'][name]:
                raise ValueError(f'Graph integrity check failed: {name}')
        self.body_ids = np.load(root / 'body_ids.npy', allow_pickle=False)
        counts = sparse.load_npz(root / 'synapse_counts.npz').tocsr()
        n = len(self.body_ids)
        if (counts.shape != (n, n) or n != self.manifest['node_count']
                or np.any(np.diff(self.body_ids) <= 0)
                or not np.isfinite(counts.data).all() or np.any(counts.data <= 0)):
            raise ValueError('Invalid graph dimensions, IDs or weights')
        # Authored normalization bounds unsigned recurrence; raw counts stay on disk.
        self._weights = counts.astype(np.float32)
        sums = np.asarray(self._weights.sum(axis=1)).ravel()
        self._weights.data /= np.repeat(np.maximum(sums, 1), np.diff(self._weights.indptr))
        self.dt_ms, self.gain = dt_ms, gain
        self.workers = min(workers, n)
        self._pool = None
        self._closed = False
        if self.workers > 1:
            cuts = np.linspace(0, n, self.workers + 1, dtype=int)
            self._parts = [self._weights[a:b] for a, b in zip(cuts[:-1], cuts[1:])]
            self._pool = ThreadPoolExecutor(max_workers=self.workers)
        self.reset()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Session is closed')
        return self

    def __exit__(self, *exc):
        self.close()

    def select(self, *, cell_type: str | None = None,
               superclass: str | None = None) -> list[int]:
        """Exact official annotation matching; combined filters use AND."""
        if not cell_type and not superclass:
            raise ValueError('Specify a cell_type or superclass')
        with (self._graph_dir / 'nodes.csv').open(encoding='utf-8', newline='') as stream:
            return [int(row['bodyId']) for row in csv.DictReader(stream)
                    if (cell_type is None or row['type'] == cell_type)
                    and (superclass is None or row['superclass'] == superclass)]

    def _indices(self, body_ids) -> np.ndarray:
        ids = np.asarray(list(body_ids))
        if ids.size == 0:
            return np.empty(0, dtype=np.int64)
        if ids.dtype.kind not in 'iu':
            raise ValueError('Body IDs must be integers')
        idx = np.searchsorted(self.body_ids, ids)
        if np.any(idx >= len(self.body_ids)) or not np.array_equal(self.body_ids[idx], ids):
            raise ValueError('Unknown or excluded body ID')
        return idx

    @property
    def state(self) -> RuntimeState:
        return RuntimeState(self._state.t_ms, self._state.activity.copy())

    def reset(self) -> None:
        self._state = RuntimeState(0.0, np.zeros(len(self.body_ids), dtype=np.float32))

    def step(self, currents: dict[int, float]) -> RuntimeState:
        if self._closed:
            raise RuntimeError('Session is closed')
        drive = np.zeros(len(self.body_ids), dtype=np.float32)
        if currents:
            idx = self._indices(currents)
            values = np.asarray(list(currents.values()), dtype=np.float64)
            if not np.isfinite(values).all() or np.any(np.abs(values) > 100):
                raise ValueError('Currents must be finite, with absolute magnitude <= 100 a.u.')
            drive[idx] = values
        activity = self._state.activity
        if self._pool is None:
            recurrent = self._weights @ activity
        else:
            recurrent = np.concatenate(list(self._pool.map(lambda part: part @ activity,
                                                           self._parts)))
        target = np.tanh(self.gain * recurrent + drive)
        updated = activity + (self.dt_ms / 80.0) * (target - activity)
        self._state = RuntimeState(self._state.t_ms + self.dt_ms, updated)
        return self.state

    def read(self, body_ids) -> dict[int, float]:
        ids = list(body_ids)
        return dict(zip(ids, map(float, self._state.activity[self._indices(ids)]), strict=True))

    def truth_metadata(self) -> dict:
        return {**self.manifest, 'dynamics': 'synthetic-unsigned-row-normalized-leaky-rate-v1',
                'dt_ms': self.dt_ms, 'tau_ms': 80, 'gain': self.gain,
                'cpu_workers': self.workers,
                'initial_state': 'zero; deterministic; no random sampling',
                'input': 'explicit body-ID currents, arbitrary units',
                'neurotransmitter_signs': 'not modeled', 'trained_readout': False,
                'claim_boundary': 'real wiring + synthetic dynamics; not biological validation'}

    def record(self, inputs, *, readout_ids, out: str | Path,
               max_steps: int = 10000) -> Path:
        """Record a bounded, application-supplied current sequence from reset.

        Only selected cells are exported; the full graph is computed. An invalid
        later input may leave the session advanced; reset before retrying.
        No files are written until all steps finish successfully.
        """
        from itertools import islice

        if self._closed:
            raise RuntimeError('Session is closed')
        if self._state.t_ms != 0:
            raise ValueError('Reset the session before recording')
        if type(max_steps) is not int or not 1 <= max_steps <= 10000:
            raise ValueError('max_steps must be an integer in [1, 10000]')
        ids = list(readout_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Specify nonempty, unique readout IDs')
        self._indices(ids)
        destination = Path(out)
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise FileExistsError('Use a fresh output directory')
        sequence = list(islice(inputs, max_steps + 1))
        if not sequence or len(sequence) > max_steps:
            raise ValueError('Input sequence is empty or exceeds max_steps')
        rows, clocks, recorded_inputs = [], [], []
        for currents in sequence:
            start_ms = self._state.t_ms
            state = self.step(currents)
            clocks.append(state.t_ms)
            rows.append(list(self.read(ids).values()))
            recorded_inputs.append({'start_ms': start_ms, 'end_ms': state.t_ms,
                                    'currents_au': {str(k): float(v) for k, v in currents.items()}})
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / 'neural_trace.npz', activity=np.asarray(rows),
                            t_ms=clocks, body_ids=np.asarray(ids, dtype=np.int64))
        (destination / 'inputs.json').write_text(json.dumps(recorded_inputs), encoding='utf-8')
        metadata = {**self.truth_metadata(), 'samples': len(rows), 'duration_ms': clocks[-1],
                    'readout_body_ids': ids, 'sampled_not_full_state_export': True,
                    'signal_units': 'arbitrary rate units', 'time_units': 'ms',
                    'artifacts': {name: sha256(destination / name)
                                  for name in ('inputs.json', 'neural_trace.npz')},
                    'graph_artifacts': self.manifest['artifacts']}
        (destination / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.out), indent=2))


if __name__ == '__main__':
    main()
