"""Runnable CPU example; synthetic graph, held-out episodes, no dataset download."""

import argparse
import json

import numpy as np

from cnskit import ConnectomeModel, Episode, Graph, Trainer, load_policy


def make_episodes(prefix, count, seed):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        x = rng.normal(size=(32, 2)).astype(np.float32)
        y = np.zeros((32, 1), dtype=np.float32)
        for t in range(32):
            y[t] = 0.6 * (y[t - 1] if t else 0) + 0.4 * (x[t, 0] - 0.3 * x[t, 1])
        result.append(Episode(f"{prefix}-{i}", x, y))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/sequence-policy")
    p.add_argument("--epochs", type=int, default=30)
    a = p.parse_args()
    graph = Graph.synthetic(32, 0.12, 7)
    model = ConnectomeModel(
        graph,
        input_ids=list(range(8)),
        readout_ids=list(range(32)),
        input_size=2,
        output_size=1,
        leak=0.4,
        seed=7,
    )
    train, valid, test = (
        make_episodes("train", 12, 1),
        make_episodes("validation", 4, 2),
        make_episodes("test", 4, 3),
    )
    trainer = Trainer(model, lr=0.02, tbptt=32, seed=7)
    report = trainer.fit(train, valid, epochs=a.epochs)
    report["test"] = trainer.evaluate(test)
    # Baseline sees exactly the same current observation, trained on the same split.
    from cns_tinker.readout import RidgeReadout

    baseline = RidgeReadout.fit(
        np.concatenate([e.observations for e in train]),
        np.concatenate([e.targets for e in train]),
        feature_names=["x0", "x1"],
        output_names=["y"],
    )
    report["memoryless_ridge_test"] = baseline.evaluate(
        np.concatenate([e.observations for e in test]),
        np.concatenate([e.targets for e in test]),
        feature_names=["x0", "x1"],
    )
    model.save(a.out, report=report)
    loaded = load_policy(a.out, expected_graph=graph.fingerprint)
    np.testing.assert_array_equal(
        model.predict(test[0].observations[None])[0], loaded.predict(test[0].observations[None])[0]
    )
    print(json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
