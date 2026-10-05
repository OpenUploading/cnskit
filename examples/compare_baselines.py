"""Matched temporal information and splits; one-seed engineering comparison only."""

import copy
import json

import numpy as np
import torch
from scipy import sparse
from torch import nn
from train_sequence import make_episodes

from cnskit import ConnectomeModel, Graph, Trainer


def main():
    torch.set_num_threads(4)
    train, valid, test = (
        make_episodes("train", 12, 1),
        make_episodes("validation", 4, 2),
        make_episodes("test", 4, 3),
    )
    graph = Graph.synthetic(32, 0.12, 7)
    results = {}
    for name, g in [
        ("synthetic_graph", graph),
        (
            "zero_edges",
            Graph(graph.body_ids, sparse.csr_matrix((32, 32)), {"dataset": "zero-edge ablation"}),
        ),
    ]:
        model = ConnectomeModel(
            g,
            input_ids=list(range(8)),
            readout_ids=list(range(32)),
            input_size=2,
            output_size=1,
            leak=0.4,
            seed=7,
        )
        trainer = Trainer(model, lr=0.02, tbptt=32, seed=7)
        report = trainer.fit(train, valid, epochs=30)
        results[name] = {
            "test_mse": trainer.evaluate(test)["loss"],
            "parameters": report["trainable_parameters"],
        }
    torch.manual_seed(7)
    rnn, head = nn.RNN(2, 5, batch_first=True), nn.Linear(5, 1)
    parameters = list(rnn.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(parameters, lr=0.02)
    values = np.concatenate([e.observations for e in train])
    mean, std = torch.tensor(values.mean(0)), torch.tensor(values.std(0))

    def loss(e):
        h, _ = rnn((torch.tensor(e.observations)[None] - mean) / std)
        return nn.functional.mse_loss(head(h)[0], torch.tensor(e.targets))

    best, checkpoint = float("inf"), None
    rng = np.random.default_rng(7)
    for _ in range(30):
        for i in rng.permutation(len(train)):
            opt.zero_grad()
            loss(train[i]).backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            opt.step()
        with torch.no_grad():
            score = sum(float(loss(e)) for e in valid) / len(valid)
        if score < best:
            best = score
            checkpoint = copy.deepcopy((rnn.state_dict(), head.state_dict()))
    rnn.load_state_dict(checkpoint[0])
    head.load_state_dict(checkpoint[1])
    with torch.no_grad():
        results["vanilla_rnn"] = {
            "test_mse": sum(float(loss(e)) for e in test) / len(test),
            "parameters": sum(p.numel() for p in parameters),
        }
    print(
        json.dumps(
            {
                "seed": 7,
                "epochs": 30,
                "results": results,
                "scope": "Synthetic task; no anatomical or model superiority claim",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
