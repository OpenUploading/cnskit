"""Classify the sign of a recent input average; synthetic data and graph."""

import argparse
import json

import numpy as np

from cnskit import ConnectomeModel, Episode, Graph, Trainer, load_policy


def dataset(prefix, seed, count):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        x = rng.normal(size=(40, 1)).astype("f")
        average = np.convolve(x[:, 0], np.ones(5) / 5, mode="full")[:40]
        labels = (average > 0).astype(np.int64)
        result.append(Episode(f"{prefix}-{i}", x, labels, np.arange(40) >= 4))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/classifier")
    args = parser.parse_args()
    graph = Graph.synthetic(32, 0.12, 7)
    model = ConnectomeModel(
        graph,
        input_ids=list(range(8)),
        readout_ids=list(range(32)),
        input_size=1,
        output_size=2,
        mode="dynamics",
        leak=0.3,
        seed=7,
        input_names=["signal"],
        output_names=["nonpositive", "positive"],
    )
    trainer = Trainer(model, objective="classification", lr=0.02, tbptt=40, seed=7)
    report = trainer.fit(dataset("train", 1, 12), dataset("validation", 2, 4), epochs=20)
    report["test"] = trainer.evaluate(dataset("test", 3, 4))
    model.save(args.out, report=report)
    loaded = load_policy(args.out)
    logits, _ = loaded.predict(dataset("inference", 4, 1)[0].observations[None])
    print(
        json.dumps(
            {
                "test": report["test"],
                "class_names": model.config["output_names"],
                "last_predicted_class": int(logits[0, -1].argmax()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
