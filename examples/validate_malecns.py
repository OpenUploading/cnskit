"""Opt-in real graph validation; synthetic targets do not imply biological fidelity."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from train_sequence import make_episodes

from cnskit import ConnectomeModel, Graph, Trainer, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.perf_counter()
    graph = Graph.from_malecns(args.graph)
    load_seconds = time.perf_counter() - start
    # Eight IDs with the largest incoming degree: deterministic engineering fixture.
    degree = np.diff(graph.weights.indptr)
    idx = np.argsort(degree, kind="stable")[-8:]
    model = ConnectomeModel(
        graph,
        input_ids=graph.body_ids[idx].tolist(),
        readout_ids=graph.body_ids[idx].tolist(),
        input_size=2,
        output_size=1,
        seed=7,
        leak=0.4,
    )
    rng = np.random.default_rng(7)
    state = rng.normal(0, 0.01, (1, len(graph.body_ids))).astype("f")
    start = time.perf_counter()
    max_error = 0.0
    with torch.no_grad():
        tensor_state = torch.from_numpy(state.copy())
        for _ in range(5):
            x = rng.normal(size=(1, 2)).astype("f")
            drive = np.zeros_like(state)
            drive[:, idx] = model.encoder(torch.from_numpy(x)).numpy()
            expected = state + 0.4 * (np.tanh(0.8 * (graph.weights @ state.T).T + drive) - state)
            _, tensor_state = model.step(x, tensor_state)
            max_error = max(max_error, float(np.abs(tensor_state.numpy() - expected).max()))
            state = expected
    full_seconds = time.perf_counter() - start
    assert max_error < 2e-6, max_error
    # Train an explicit 128-cell induced graph; retain parent normalization.
    chosen = np.argsort(degree, kind="stable")[-128:]
    sub = graph.subgraph(graph.body_ids[chosen].tolist())
    del model
    small = ConnectomeModel(
        sub,
        input_ids=sub.body_ids[:8].tolist(),
        readout_ids=sub.body_ids.tolist(),
        input_size=2,
        output_size=1,
        seed=7,
        leak=0.4,
    )
    trainer = Trainer(small, lr=0.02, tbptt=32, seed=7)
    train, valid, test = (
        make_episodes("train", 12, 1),
        make_episodes("validation", 4, 2),
        make_episodes("test", 4, 3),
    )
    before = trainer.evaluate(test)
    report = trainer.fit(train, valid, epochs=30)
    after = trainer.evaluate(test)
    small.save(root / "policy", report=report)
    restored = load_policy(root / "policy", expected_graph=sub.fingerprint)
    np.testing.assert_array_equal(
        small.predict(test[0].observations[None])[0],
        restored.predict(test[0].observations[None])[0],
    )
    result = {
        "scope": "Real anatomy; authored dynamics and synthetic temporal-filter targets",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "graph_nodes": len(graph.body_ids),
        "graph_edges": int(graph.weights.nnz),
        "graph_fingerprint": graph.fingerprint,
        "graph_load_seconds": load_seconds,
        "full_graph_steps": 5,
        "full_graph_seconds": full_seconds,
        "scipy_reference_max_abs_error": max_error,
        "training_subgraph_nodes": len(sub.body_ids),
        "training_subgraph_edges": sub.weights.nnz,
        "test_before": before,
        "test_after": after,
        "best_validation_loss": report["best_validation_loss"],
        "checkpoint_replay_exact": True,
    }
    (root / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
