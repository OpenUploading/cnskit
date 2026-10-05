import numpy as np
import pytest
import torch
from scipy import sparse

from cnskit import ConnectomeModel, Episode, Graph, Trainer, load_policy


def model(mode="adapters"):
    return ConnectomeModel(
        Graph.synthetic(8, 0.3, 4),
        input_ids=[0, 1],
        readout_ids=list(range(8)),
        input_size=2,
        output_size=1,
        mode=mode,
        leak=0.5,
    )


def episodes(prefix, n=6):
    rng = np.random.default_rng(42 if prefix == "train" else 91)
    return [
        Episode(
            f"{prefix}-{i}",
            x := rng.normal(size=(16, 2)).astype("f"),
            (0.7 * x[:, 0:1] - 0.2 * x[:, 1:2]),
        )
        for i in range(n)
    ]


def test_sparse_forward_and_adapter_gradient_match_dense_reference():
    m = model("dynamics")
    x = torch.tensor([[0.2, -0.3]])
    s = torch.arange(8, dtype=torch.float32)[None] / 20
    y, next_state = m.step(x, s)
    drive = torch.zeros_like(s).index_add(1, m.input_indices, m.encoder(x))
    dense = s + torch.sigmoid(m.leak_logit) * (
        torch.tanh(
            torch.sigmoid(m.gain_logit) * (s @ torch.tensor(m.graph.weights.toarray()).T) + drive
        )
        - s
    )
    torch.testing.assert_close(next_state, dense)
    loss = y.sum()
    loss.backward()
    analytic = m.encoder.weight.grad[0, 0].item()
    original = m.encoder.weight[0, 0].item()
    with torch.no_grad():
        m.encoder.weight[0, 0] = original + 1e-3
        plus = m.step(x, s)[0].sum().item()
        m.encoder.weight[0, 0] = original - 1e-3
        minus = m.step(x, s)[0].sum().item()
        m.encoder.weight[0, 0] = original
    assert analytic == pytest.approx((plus - minus) / 0.002, abs=2e-5)


def test_training_generalizes_and_checkpoint_replays(tmp_path):
    m = model()
    trainer = Trainer(m, lr=0.03, tbptt=16)
    train, valid, test = episodes("train"), episodes("validation", 3), episodes("test", 3)
    before = trainer.evaluate(valid)["loss"]
    report = trainer.fit(train, valid, epochs=20)
    assert trainer.evaluate(valid)["loss"] < before * 0.5
    expected = np.concatenate([e.observations for e in train]).mean(0)
    np.testing.assert_allclose(m.input_mean.numpy(), expected, atol=1e-6)
    pred, _ = m.predict(test[0].observations[None])
    m.save(tmp_path / "policy", report=report)
    restored = load_policy(tmp_path / "policy", expected_graph=m.graph.fingerprint)
    np.testing.assert_array_equal(pred, restored.predict(test[0].observations[None])[0])
    with pytest.raises(ValueError, match="graph"):
        load_policy(tmp_path / "policy", expected_graph="wrong")
    with (tmp_path / "policy" / "parameters.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="integrity"):
        load_policy(tmp_path / "policy")


def test_episode_isolation_modes_and_validation():
    m = model("readout")
    t = Trainer(m)
    train = episodes("train")
    with pytest.raises(ValueError, match="overlap"):
        t.fit(train, train)
    before = m.encoder.weight.detach().clone()
    t.fit(train, episodes("val", 2), epochs=2)
    torch.testing.assert_close(before, m.encoder.weight)
    x = torch.zeros(2, 5, 2)
    a, _ = m(x)
    b, _ = m(x[:1])
    torch.testing.assert_close(a[:1], b)
    for bad in (np.array([[np.nan, 1]]), np.zeros((1, 3))):
        with pytest.raises(ValueError):
            m.step(bad)
    with pytest.raises(ValueError):
        Episode("bad", np.zeros((3, 2)), np.zeros((3, 1)), np.zeros(3, dtype=bool)).validate(
            2, 1, "regression"
        )


def test_classification_mask_and_subgraph():
    g = Graph(
        np.array([10, 20, 30]),
        sparse.csr_matrix([[0, 0, 0], [0.5, 0, 0], [0, 0.2, 0]]),
        {"dataset": "test"},
    )
    subset = g.subgraph([20, 10])
    assert subset.body_ids.tolist() == [20, 10]
    assert subset.weights[0, 1] == 0.5
    with pytest.raises(ValueError):
        g.indices([99])
    m = ConnectomeModel(
        g, input_ids=[10], readout_ids=[20], input_size=1, output_size=2, mode="dynamics"
    )
    e = Episode(
        "one", np.ones((4, 1)), np.array([0, 1, 0, 1]), np.array([False, True, False, True])
    )
    assert Trainer(m, objective="classification").evaluate([e])["labeled_steps"] == 2
    Trainer(m, objective="classification", tbptt=2).fit(
        [e], [Episode("two", e.observations, e.targets, e.mask)], epochs=2
    )


def test_streaming_chunk_equivalence_and_channel_binding(tmp_path):
    m = model()
    x = np.random.default_rng(3).normal(size=(2, 17, 2)).astype("f")
    whole, final = m.predict(x)
    left, state = m.predict(x[:, :7])
    right, end = m.predict(x[:, 7:], state=state)
    np.testing.assert_array_equal(whole, np.concatenate([left, right], axis=1))
    torch.testing.assert_close(final, end, rtol=0, atol=0)
    assert not state.requires_grad
    assert not m.edge_values.requires_grad
    with pytest.raises(ValueError):
        ConnectomeModel(m.graph, input_ids=[0.5], readout_ids=[1], input_size=1, output_size=1)
    with pytest.raises(ValueError):
        ConnectomeModel(
            m.graph,
            input_ids=[0],
            readout_ids=[1],
            input_size=2,
            output_size=1,
            input_names=["duplicate", "duplicate"],
        )
    m.save(tmp_path / "named")
    assert load_policy(tmp_path / "named").config["input_names"] == ["x0", "x1"]


def test_cli_training_and_prediction(tmp_path):
    import json
    import subprocess
    import sys

    from test_malecns import fixture_graph

    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    fixture_graph(graph_dir)
    config = {
        "graph_dir": str(graph_dir),
        "model": {"input_ids": [10], "readout_ids": [20, 30], "input_size": 2, "output_size": 1},
        "epochs": 2,
    }
    (tmp_path / "task.json").write_text(json.dumps(config))
    for split in ["train", "validation"]:
        es = episodes(split, 2)
        np.savez(
            tmp_path / f"{split}.npz",
            observations=np.stack([e.observations for e in es]),
            targets=np.stack([e.targets for e in es]),
            episode_ids=[e.id for e in es],
        )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cnskit.cli",
            "train",
            "--config",
            str(tmp_path / "task.json"),
            "--train",
            str(tmp_path / "train.npz"),
            "--validation",
            str(tmp_path / "validation.npz"),
            "--out",
            str(tmp_path / "policy"),
        ],
        check=True,
        capture_output=True,
    )
    x = episodes("input", 1)[0].observations[None]
    np.save(tmp_path / "input.npy", x)
    command = [
        sys.executable,
        "-m",
        "cnskit.cli",
        "predict",
        "--policy",
        str(tmp_path / "policy"),
        "--input",
        str(tmp_path / "input.npy"),
        "--out",
        str(tmp_path / "prediction.npy"),
    ]
    subprocess.run(command, check=True, capture_output=True)
    np.testing.assert_array_equal(
        np.load(tmp_path / "prediction.npy"), load_policy(tmp_path / "policy").predict(x)[0]
    )
    assert subprocess.run(command, capture_output=True).returncode != 0
