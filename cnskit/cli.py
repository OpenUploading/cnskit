"""Offline sequence training and inference without a server or account."""

import argparse
import json
from pathlib import Path

import numpy as np

from .graph import Graph


def load_episodes(path):
    from .learning import Episode

    with np.load(path, allow_pickle=False) as data:
        x, y, ids = data["observations"], data["targets"], data["episode_ids"]
        if (
            x.ndim != 3
            or y.ndim not in (2, 3)
            or len(x) != len(y)
            or ids.shape != (len(x),)
            or ids.dtype.kind not in "US"
        ):
            raise ValueError(
                "Expected observations [episode,time,input], aligned targets and string episode_ids"
            )
        masks = data["mask"] if "mask" in data else np.ones(x.shape[:2], dtype=bool)
        if masks.shape != x.shape[:2]:
            raise ValueError("Mask dimensions do not match episodes")
        return [Episode(str(ids[i]), x[i], y[i], masks[i]) for i in range(len(x))]


def main():
    p = argparse.ArgumentParser(prog="cnskit", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--train", required=True)
    train.add_argument("--validation", required=True)
    train.add_argument("--out", required=True)
    train.add_argument("--device", default="cpu")
    predict = sub.add_parser("predict")
    predict.add_argument("--policy", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--out", required=True)
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--chunk-size", type=int, default=1024)
    evaluate = sub.add_parser("evaluate", help="Evaluate a held-out episode dataset")
    evaluate.add_argument("--policy", required=True)
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--objective", choices=["regression", "classification"], required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--chunk-size", type=int, default=1024)
    a = p.parse_args()
    from .learning import ConnectomeModel, Trainer, load_policy

    if a.command != "evaluate" and Path(a.out).exists():
        p.error("Output already exists; choose a fresh path")
    if a.command == "train":
        cfg = json.loads(Path(a.config).read_text(encoding="utf8"))
        graph = Graph.from_malecns(cfg["graph_dir"], body_ids=cfg.get("subgraph_ids"))
        model = ConnectomeModel(graph, **cfg["model"]).to(a.device)
        trainer = Trainer(model, **cfg.get("trainer", {}))
        report = trainer.fit(
            load_episodes(a.train),
            load_episodes(a.validation),
            epochs=cfg.get("epochs", 30),
            patience=cfg.get("patience"),
            min_delta=cfg.get("min_delta", 0.0),
        )
        model.save(a.out, report=report)
        print(json.dumps({"best_validation_loss": report["best_validation_loss"], "policy": a.out}))
    elif a.command == "evaluate":
        model = load_policy(a.policy, device=a.device)
        metrics = Trainer(model, objective=a.objective).evaluate(
            load_episodes(a.data), chunk_size=a.chunk_size
        )
        print(json.dumps(metrics, allow_nan=False))
    else:
        from .inference import predict_file

        model = load_policy(a.policy, device=a.device)
        report = predict_file(model, a.input, a.out, chunk_size=a.chunk_size)
        print(json.dumps(report))


if __name__ == "__main__":
    main()
