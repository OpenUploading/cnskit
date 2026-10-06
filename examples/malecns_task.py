"""Real MaleCNS anatomy, explicit task data, held-out evaluation and portable inference.

Without dataset arguments, targets are synthetic temporal-filter targets, not behavior.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from train_sequence import make_episodes

from cnskit import ConnectomeModel, Graph, Trainer, load_policy
from cnskit.cli import load_episodes


def save_episodes(path, episodes):
    np.savez_compressed(
        path,
        observations=np.stack([e.observations for e in episodes]),
        targets=np.stack([e.targets for e in episodes]),
        episode_ids=np.array([e.id for e in episodes]),
        mask=np.stack(
            [
                np.ones(len(e.observations), dtype=bool) if e.mask is None else e.mask
                for e in episodes
            ]
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, help="Prepared, hash-verified MaleCNS directory")
    p.add_argument("--out", required=True, help="New output directory")
    p.add_argument("--train")
    p.add_argument("--validation")
    p.add_argument("--test")
    p.add_argument("--cells", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    args = p.parse_args()
    supplied = [args.train, args.validation, args.test]
    if any(supplied) and not all(supplied):
        p.error("Supply all three split files, or none for the generated regression task")
    if args.cells < 8 or args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        p.error("Require at least eight cells, positive epochs and unique seeds")
    root = Path(args.out)
    if root.exists():
        p.error("Output exists; choose a fresh directory")
    torch.set_num_threads(4)
    full = Graph.from_malecns(args.graph)
    if args.cells > len(full.body_ids):
        p.error("Requested more cells than the graph contains")
    # Top incoming degree is reproducible, not a task-specific biological circuit claim.
    chosen = np.argsort(np.diff(full.weights.indptr), kind="stable")[-args.cells :]
    graph = full.subgraph(full.body_ids[chosen].tolist())
    del full
    if all(supplied):
        train, valid, test = [load_episodes(path) for path in supplied]
    else:
        train, valid, test = [
            make_episodes(name, count, seed)
            for name, count, seed in [("train", 12, 1), ("validation", 4, 2), ("test", 4, 3)]
        ]
    id_sets = [{e.id for e in split} for split in (train, valid, test)]
    if any(id_sets[i] & id_sets[j] for i, j in [(0, 1), (0, 2), (1, 2)]):
        p.error("Train, validation and test episode IDs must be disjoint")
    inputs = np.asarray(train[0].observations).shape[-1]
    target = np.asarray(train[0].targets)
    if target.ndim != 2:
        p.error("This example expects regression targets [time, output]")
    outputs = target.shape[-1]
    for split in (train, valid, test):
        for episode in split:
            episode.validate(inputs, outputs, "regression")
    root.mkdir(parents=True)
    if not all(supplied):
        for name, split in [("train", train), ("validation", valid), ("test", test)]:
            save_episodes(root / f"{name}.npz", split)
    config = dict(
        input_ids=graph.body_ids[:8].tolist(),
        readout_ids=graph.body_ids.tolist(),
        input_size=inputs,
        output_size=outputs,
        mode="adapters",
        leak=0.4,
    )
    task = {
        "graph_dir": str(Path(args.graph).resolve()),
        "subgraph_ids": graph.body_ids.tolist(),
        "model": {**config, "seed": args.seeds[0]},
        "trainer": {"lr": 0.02, "tbptt": 32, "seed": args.seeds[0]},
        "epochs": args.epochs,
    }
    (root / "task.json").write_text(json.dumps(task, indent=2), encoding="utf8")
    results = []
    zero = Graph(graph.body_ids, sparse.csr_matrix(graph.weights.shape), {"ablation": "zero edges"})
    for seed in args.seeds:
        for label, selected in [("anatomical", graph), ("zero_edges", zero)]:
            model = ConnectomeModel(selected, **config, seed=seed)
            trainer = Trainer(model, lr=0.02, tbptt=32, seed=seed)
            report = trainer.fit(train, valid, epochs=args.epochs)
            metrics = trainer.evaluate(test)
            row = {
                "seed": seed,
                "model": label,
                "test": metrics,
                "best_epoch": report["best_epoch"],
                "trainable_parameters": report["trainable_parameters"],
            }
            results.append(row)
            print(json.dumps(row), flush=True)
            # Export the first requested seed, not the seed with the best test score.
            if seed == args.seeds[0] and label == "anatomical":
                model.save(root / "policy", report=report)
                restored = load_policy(root / "policy", expected_graph=graph.fingerprint)
                x = test[0].observations[None]
                predicted, _ = model.predict(x)
                replay, _ = restored.predict(x)
                np.testing.assert_array_equal(predicted, replay)
                midpoint = max(1, x.shape[1] // 2)
                left, state = restored.predict(x[:, :midpoint])
                if midpoint < x.shape[1]:
                    right, _ = restored.predict(x[:, midpoint:], state=state)
                    np.testing.assert_array_equal(replay, np.concatenate([left, right], axis=1))
                np.save(root / "observations.npy", x)
                np.save(root / "predictions.npy", replay)
    summary = {
        "task": "user regression episodes" if all(supplied) else "synthetic temporal filtering",
        "selection": "top incoming degree; parent weight normalization retained",
        "nodes": len(graph.body_ids),
        "edges": graph.weights.nnz,
        "fingerprint": graph.fingerprint,
        "checkpoint_replay_exact": True,
        "results": results,
        "mean_test_mse": {
            label: float(np.mean([row["test"]["loss"] for row in results if row["model"] == label]))
            for label in ("anatomical", "zero_edges")
        },
        "scope": "Engineering experiment; not measured behavior or a general reasoning benchmark",
    }
    (root / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary["mean_test_mse"], indent=2))


if __name__ == "__main__":
    main()
