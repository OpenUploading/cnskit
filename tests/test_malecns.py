"""Tiny artificial fixtures test implementation, not biological accuracy."""
import json

import numpy as np
import pytest

pytest.importorskip('scipy')
from scipy import sparse

from cns_tinker.runtime.malecns import MaleCNSSession, sha256


def fixture_graph(root):
    np.save(root / 'body_ids.npy', np.array([10, 20, 30], dtype=np.int64))
    # 10 -> 20; cell 30 disconnected. Rows are receivers.
    sparse.save_npz(root / 'synapse_counts.npz', sparse.csr_matrix([[0, 0, 0], [3, 0, 0], [0, 0, 0]]))
    (root / 'nodes.csv').write_text('bodyId,type,superclass\n10,input,sensory\n20,output,motor\n30,other,motor\n')
    manifest = {'dataset': 'male-cns:v1.0', 'node_count': 3,
                'test_fixture': True, 'artifacts': {name: sha256(root / name) for name in
                 ('body_ids.npy', 'synapse_counts.npz', 'nodes.csv')}}
    (root / 'manifest.json').write_text(json.dumps(manifest))


def test_edges_carry_signal_in_correct_direction(tmp_path):
    fixture_graph(tmp_path)
    run = MaleCNSSession(tmp_path)
    for _ in range(20):
        run.step({10: 1})
    assert run.read([20])[20] > 0
    assert run.read([30])[30] == 0
    expected = run.state.activity
    run.reset()
    for _ in range(20):
        run.step({10: 1})
    np.testing.assert_array_equal(expected, run.state.activity)
    run.reset()
    for _ in range(20):
        run.step({20: 1})
    assert run.read([10])[10] == 0


def test_unknown_ids_invalid_inputs_and_integrity_fail_closed(tmp_path):
    fixture_graph(tmp_path)
    run = MaleCNSSession(tmp_path)
    for currents in ({99: 1}, {10: float('nan')}, {10.5: 1}, {10: 101}):
        with pytest.raises(ValueError):
            run.step(currents)
        assert run.state.t_ms == 0
    assert run.read([]) == {}
    state = run.state
    state.activity[:] = 1
    assert not run.state.activity.any()
    (tmp_path / 'nodes.csv').write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        MaleCNSSession(tmp_path)


def test_import_preserves_counts_and_excludes_untraced_nodes(tmp_path):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.feather as feather
    from cns_tinker.runtime.malecns import ANNOTATIONS, WEIGHTS, prepare

    raw = tmp_path / 'raw'; raw.mkdir()
    feather.write_feather(pa.table({'bodyId': [20, 10, 30],
        'status': ['Traced', 'Traced', 'Orphan'], 'type': ['B', 'A', 'C'],
        'instance': ['', '', ''], 'superclass': ['', '', '']}), raw / ANNOTATIONS)
    feather.write_feather(pa.table({'body_pre': [10, 10, 30],
        'body_post': [20, 20, 10], 'weight': [3, 2, 7]}), raw / WEIGHTS)
    meta = prepare(raw, tmp_path / 'graph')
    assert meta['node_count'] == 2 and meta['edge_count'] == 1
    assert meta['synapse_count'] == 5
    matrix = sparse.load_npz(tmp_path / 'graph/synapse_counts.npz').toarray()
    np.testing.assert_array_equal(matrix, [[0, 0], [5, 0]])


def test_parallel_record_matches_serial_and_preserves_inputs(tmp_path):
    graph = tmp_path / 'graph'; graph.mkdir(); fixture_graph(graph)
    sequence = [{10: 1.0}] * 10 + [{}] * 10
    with MaleCNSSession(graph) as serial, MaleCNSSession(graph, workers=3) as parallel:
        assert parallel.select(cell_type='input') == [10]
        assert parallel.select(superclass='motor', cell_type='output') == [20]
        assert parallel.select(cell_type='absent') == []
        for currents in sequence:
            np.testing.assert_array_equal(serial.step(currents).activity,
                                          parallel.step(currents).activity)
        parallel.reset()
        out = parallel.record(sequence, readout_ids=[20], out=tmp_path/'run')
        trace = np.load(out/'neural_trace.npz')
        assert trace['activity'].shape == (20, 1)
        assert trace['t_ms'][-1] == 100
        np.testing.assert_array_equal(trace['activity'][-1], [serial.read([20])[20]])
        inputs = json.loads((out/'inputs.json').read_text())
        assert inputs[0] == {'start_ms': 0, 'end_ms': 5, 'currents_au': {'10': 1.0}}
        assert inputs[-1]['currents_au'] == {}
    with pytest.raises(RuntimeError, match='closed'):
        parallel.step({})


def test_record_rejects_bad_requests_without_advancing(tmp_path):
    fixture_graph(tmp_path)
    with MaleCNSSession(tmp_path) as run:
        for inputs, ids in [([], [10]), ([{}], []), ([{}], [99]), ([{}], [10,10])]:
            with pytest.raises(ValueError):
                run.record(inputs, readout_ids=ids, out=tmp_path/'run')
            assert run.state.t_ms == 0
        with pytest.raises(ValueError, match='exceeds'):
            run.record([{}]*3, readout_ids=[10], out=tmp_path/'run', max_steps=2)
        assert run.state.t_ms == 0
        assert not (tmp_path/'run').exists()


def test_real_run_cli_uses_custom_inputs(tmp_path):
    from cns_tinker.cli import main

    graph = tmp_path/'graph'; graph.mkdir(); fixture_graph(graph)
    source = tmp_path/'input.json'
    source.write_text(json.dumps([{'10': 1.0}]*8 + [{}]*4))
    out = tmp_path/'recording'
    assert main(['real-run', '--graph', str(graph), '--inputs', str(source),
                 '--readout-type', 'output', '--workers', '2', '--out', str(out)]) == 0
    trace = np.load(out/'neural_trace.npz')
    assert trace['activity'].shape == (12,1)
    assert trace['activity'].max() > 0
    manifest = json.loads((out/'manifest.json').read_text())
    assert manifest['cpu_workers'] == 2
    assert manifest['readout_body_ids'] == [20]
    for name, digest in manifest['artifacts'].items():
        assert sha256(out/name) == digest
