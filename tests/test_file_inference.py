import numpy as np
import pytest

from cnskit import ConnectomeModel, Graph
from cnskit.inference import predict_file


def policy():
    return ConnectomeModel(
        Graph.synthetic(8, 0.3, 7),
        input_ids=[0, 1],
        readout_ids=list(range(8)),
        input_size=2,
        output_size=3,
    )


@pytest.mark.parametrize("fortran", [False, True])
def test_file_prediction_matches_episode_inference_and_bounds_buffers(tmp_path, fortran):
    m = policy()
    x = np.random.default_rng(7).normal(size=(3, 19, 2)).astype("f")
    np.save(tmp_path / "input.npy", np.asfortranarray(x) if fortran else x)
    expected = np.concatenate([m.predict(row[None])[0] for row in x])
    calls = []
    hook = m.register_forward_pre_hook(lambda module, args: calls.append(args[0].shape))
    try:
        report = predict_file(m, tmp_path / "input.npy", tmp_path / "out.npy", chunk_size=4)
    finally:
        hook.remove()
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), expected)
    assert report["episodes"] == 3
    assert all(shape[0] == 1 and shape[1] <= 4 for shape in calls)
    with pytest.raises(FileExistsError):
        predict_file(m, tmp_path / "input.npy", tmp_path / "out.npy")
    np.testing.assert_array_equal(np.load(tmp_path / "out.npy"), expected)
    assert not list(tmp_path.glob(".cnskit-predict-*"))


def test_failed_late_chunk_does_not_publish_partial_output(tmp_path):
    x = np.ones((2, 11, 2), dtype="f")
    x[1, -1, 0] = np.nan
    np.save(tmp_path / "input.npy", x)
    with pytest.raises(ValueError):
        predict_file(policy(), tmp_path / "input.npy", tmp_path / "out.npy", chunk_size=3)
    assert not (tmp_path / "out.npy").exists()
    assert not list(tmp_path.glob(".cnskit-predict-*"))


def test_concurrent_destination_is_preserved(tmp_path, monkeypatch):
    import cnskit.inference as inference

    np.save(tmp_path / "input.npy", np.ones((1, 2, 2), dtype="f"))
    link = inference.os.link

    def race(source, destination):
        destination.write_bytes(b"other writer")
        link(source, destination)

    monkeypatch.setattr(inference.os, "link", race)
    with pytest.raises(FileExistsError):
        predict_file(policy(), tmp_path / "input.npy", tmp_path / "out.npy")
    assert (tmp_path / "out.npy").read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".cnskit-predict-*"))
