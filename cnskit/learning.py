"""Differentiable sparse recurrence, episode training and portable inference.

The anatomical graph is fixed. Modes select a readout, input adapter, or bounded
cell-wise gain/leak adaptation. Adapted dynamics are learned models, not anatomy.
"""

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import nn

from .graph import Graph


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Episode:
    id: str
    observations: np.ndarray
    targets: np.ndarray
    mask: np.ndarray | None = None

    def validate(self, inputs, outputs, objective):
        x, y = np.asarray(self.observations), np.asarray(self.targets)
        if (
            not isinstance(self.id, str)
            or not self.id
            or x.ndim != 2
            or x.shape[1] != inputs
            or not len(x)
            or not np.isfinite(x).all()
        ):
            raise ValueError("Invalid episode ID or observations")
        if objective == "regression":
            if y.shape != (len(x), outputs) or not np.isfinite(y).all():
                raise ValueError("Regression targets must be finite [time, outputs]")
        elif (
            y.shape != (len(x),)
            or y.dtype.kind not in "iu"
            or np.any(y < 0)
            or np.any(y >= outputs)
        ):
            raise ValueError("Classification targets must be integer class IDs [time]")
        mask = np.ones(len(x), dtype=bool) if self.mask is None else np.asarray(self.mask)
        if mask.shape != (len(x),) or mask.dtype != bool or not mask.any():
            raise ValueError("Mask must select at least one timestep")
        return x, y, mask


class ConnectomeModel(nn.Module):
    def __init__(
        self,
        graph: Graph,
        *,
        input_ids,
        readout_ids,
        input_size,
        output_size,
        mode="adapters",
        gain=0.8,
        leak=0.0625,
        seed=0,
        input_names=None,
        output_names=None,
    ):
        super().__init__()
        if mode not in ("readout", "adapters", "dynamics"):
            raise ValueError("mode must be readout, adapters or dynamics")
        if (
            type(input_size) is not int
            or input_size < 1
            or type(output_size) is not int
            or output_size < 1
        ):
            raise ValueError("Input/output dimensions must be positive integers")
        if not 0 < gain < 1 or not 0 < leak < 1:
            raise ValueError("gain and leak must be in (0,1)")

        def channels(names, count, prefix):
            names = list(names) if names is not None else [f"{prefix}{i}" for i in range(count)]
            if (
                len(names) != count
                or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != count
            ):
                raise ValueError("Channel names must be nonempty, unique and match dimensions")
            return names

        input_ids, readout_ids = list(input_ids), list(readout_ids)
        graph.indices(input_ids)
        graph.indices(readout_ids)
        self.graph = graph = Graph(graph.body_ids, graph.weights, graph.provenance)
        self.config = dict(
            input_ids=list(map(int, input_ids)),
            readout_ids=list(map(int, readout_ids)),
            input_size=input_size,
            output_size=output_size,
            mode=mode,
            gain=gain,
            leak=leak,
            seed=seed,
            input_names=channels(input_names, input_size, "x"),
            output_names=channels(output_names, output_size, "y"),
        )
        self.register_buffer("input_indices", torch.tensor(graph.indices(self.config["input_ids"])))
        self.register_buffer(
            "readout_indices", torch.tensor(graph.indices(self.config["readout_ids"]))
        )
        coo = graph.weights.tocoo()
        self.register_buffer("edges", torch.tensor(np.stack([coo.row, coo.col]), dtype=torch.long))
        self.register_buffer("edge_values", torch.tensor(coo.data))
        self.register_buffer(
            "_operator",
            torch.sparse_coo_tensor(
                self.edges, self.edge_values, graph.weights.shape, check_invariants=True
            ).coalesce(),
            persistent=False,
        )
        self.register_buffer("input_mean", torch.zeros(input_size))
        self.register_buffer("input_scale", torch.ones(input_size))
        # Avoid resetting the caller's global random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = nn.Linear(input_size, len(self.input_indices))
            self.decoder = nn.Linear(len(self.readout_indices), output_size)
        self.encoder.requires_grad_(mode != "readout")
        n = len(graph.body_ids)
        self.gain_logit = nn.Parameter(
            torch.full((n,), math.log(gain / (1 - gain))), requires_grad=mode == "dynamics"
        )
        self.leak_logit = nn.Parameter(
            torch.full((n,), math.log(leak / (1 - leak))), requires_grad=mode == "dynamics"
        )

    @property
    def device(self):
        return self.input_mean.device

    def initial_state(self, batch_size=1):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        return torch.zeros(batch_size, len(self.graph.body_ids), device=self.device)

    def step(self, observation, state=None):
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if (
            x.ndim != 2
            or not len(x)
            or x.shape[1] != self.config["input_size"]
            or not torch.isfinite(x).all()
        ):
            raise ValueError("Expected finite [batch, input_size] observations")
        if state is None:
            state = self.initial_state(len(x))
        if (
            state.shape != (len(x), len(self.graph.body_ids))
            or state.device != self.device
            or state.dtype != torch.float32
            or not torch.isfinite(state).all()
        ):
            raise ValueError("State shape/device/values do not match model")
        drive = torch.zeros_like(state).index_add(
            1, self.input_indices, self.encoder((x - self.input_mean) / self.input_scale)
        )
        recurrent = torch.sparse.mm(self._operator, state.T).T
        target = torch.tanh(torch.sigmoid(self.gain_logit) * recurrent + drive)
        next_state = state + torch.sigmoid(self.leak_logit) * (target - state)
        return self.decoder(next_state.index_select(1, self.readout_indices)), next_state

    def forward(self, observations, state=None):
        x = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        if x.ndim != 3 or x.shape[1] == 0:
            raise ValueError("Expected [batch, time, input_size]")
        outputs = []
        for t in range(x.shape[1]):
            y, state = self.step(x[:, t], state)
            outputs.append(y)
        return torch.stack(outputs, dim=1), state

    @torch.no_grad()
    def predict(self, observations, *, state=None):
        """Explicit state: reset with None between independent episodes."""
        self.eval()
        output, state = self(observations, state)
        return output.cpu().numpy(), state.detach()

    def save(self, path, *, report=None):
        root = Path(path)
        root.mkdir(parents=True, exist_ok=False)
        sparse.save_npz(root / "graph.npz", self.graph.weights)
        np.save(root / "body_ids.npy", self.graph.body_ids, allow_pickle=False)
        np.savez_compressed(
            root / "parameters.npz",
            **{k: v.detach().cpu().numpy() for k, v in self.state_dict().items()},
        )
        metadata = {
            "schema": "cnskit.policy.v1",
            "config": self.config,
            "graph_fingerprint": self.graph.fingerprint,
            "provenance": self.graph.provenance,
            "report": report,
            "files": {
                name: _file_hash(root / name)
                for name in ("graph.npz", "body_ids.npy", "parameters.npz")
            },
        }
        (root / "manifest.json").write_text(
            json.dumps(metadata, indent=2, allow_nan=False), encoding="utf8"
        )


def load_policy(path, *, device="cpu", expected_graph=None):
    root = Path(path)
    m = json.loads((root / "manifest.json").read_text(encoding="utf8"))
    if m.get("schema") != "cnskit.policy.v1":
        raise ValueError("Unsupported policy format")
    for name in ("graph.npz", "body_ids.npy", "parameters.npz"):
        if _file_hash(root / name) != m["files"][name]:
            raise ValueError(f"Checkpoint integrity failed: {name}")
    graph = Graph(
        np.load(root / "body_ids.npy", allow_pickle=False),
        sparse.load_npz(root / "graph.npz"),
        m["provenance"],
    )
    if graph.fingerprint != m["graph_fingerprint"] or (
        expected_graph is not None and graph.fingerprint != expected_graph
    ):
        raise ValueError("Checkpoint graph does not match expected graph")
    model = ConnectomeModel(graph, **m["config"])
    with np.load(root / "parameters.npz", allow_pickle=False) as data:
        state = {k: torch.from_numpy(data[k].copy()) for k in data.files}
    expected = model.state_dict()
    if state.keys() != expected.keys() or any(
        state[k].shape != v.shape or state[k].dtype != v.dtype or not torch.isfinite(state[k]).all()
        for k, v in expected.items()
    ):
        raise ValueError("Invalid checkpoint parameter schema")
    for k in ("edges", "edge_values", "input_indices", "readout_indices"):
        if not torch.equal(state[k], expected[k]):
            raise ValueError("Checkpoint graph binding mismatch")
    if torch.any(state["input_scale"] <= 0):
        raise ValueError("Invalid normalization scale")
    model.load_state_dict(state)
    return model.to(device).eval()


class Trainer:
    def __init__(self, model, *, objective="regression", lr=0.01, tbptt=64, grad_clip=1.0, seed=0):
        if (
            objective not in ("regression", "classification")
            or not math.isfinite(lr)
            or lr <= 0
            or type(tbptt) is not int
            or tbptt < 1
            or not math.isfinite(grad_clip)
            or grad_clip <= 0
        ):
            raise ValueError("Invalid training configuration")
        self.model, self.objective, self.lr, self.tbptt, self.grad_clip, self.seed = (
            model,
            objective,
            lr,
            tbptt,
            grad_clip,
            seed,
        )

    def _checked(self, episodes):
        items = list(episodes)
        if not items or len({e.id for e in items}) != len(items):
            raise ValueError("Episodes must have unique IDs and be nonempty")
        for e in items:
            e.validate(
                self.model.config["input_size"], self.model.config["output_size"], self.objective
            )
        return items

    def _loss(self, prediction, target):
        return (
            nn.functional.mse_loss(prediction, target)
            if self.objective == "regression"
            else nn.functional.cross_entropy(prediction, target)
        )

    @torch.no_grad()
    def evaluate(self, episodes):
        items = self._checked(episodes)
        total = 0.0
        count = 0
        correct = 0
        self.model.eval()
        for e in items:
            x, y, mask = e.validate(
                self.model.config["input_size"], self.model.config["output_size"], self.objective
            )
            p, _ = self.model(x[None])
            p = p[0, torch.tensor(mask, device=self.model.device)]
            target = torch.as_tensor(
                y[mask],
                device=self.model.device,
                dtype=torch.float32 if self.objective == "regression" else torch.long,
            )
            loss = float(self._loss(p, target))
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite evaluation loss")
            total += loss * int(mask.sum())
            count += int(mask.sum())
            if self.objective == "classification":
                correct += int((p.argmax(-1) == target).sum())
        result = {"loss": total / count, "labeled_steps": count, "episodes": len(items)}
        if self.objective == "classification":
            result["accuracy"] = correct / count
        return result

    def fit(self, train, validation, *, epochs=30):
        train, validation = self._checked(train), self._checked(validation)
        if {e.id for e in train} & {e.id for e in validation}:
            raise ValueError("Training and validation episode IDs overlap")
        if type(epochs) is not int or epochs < 1:
            raise ValueError("epochs must be positive")
        # Fit normalization only on training episodes, with bounded temporary memory.
        count = 0
        mean = np.zeros(self.model.config["input_size"])
        m2 = np.zeros_like(mean)
        for e in train:
            x = np.asarray(e.observations, dtype=np.float64)
            n = len(x)
            delta = x.mean(0) - mean
            m2 += ((x - x.mean(0)) ** 2).sum(0) + delta**2 * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
        scale = np.sqrt(m2 / count)
        scale = np.where(scale > 1e-6, scale, 1.0)
        self.model.input_mean.copy_(
            torch.tensor(mean, dtype=torch.float32, device=self.model.device)
        )
        self.model.input_scale.copy_(
            torch.tensor(scale, dtype=torch.float32, device=self.model.device)
        )
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        rng = np.random.default_rng(self.seed)
        best = float("inf")
        best_state = None
        history = []
        start = time.perf_counter()
        for epoch in range(epochs):
            self.model.train()
            for index in rng.permutation(len(train)):
                e = train[index]
                x, y, mask = e.validate(
                    self.model.config["input_size"],
                    self.model.config["output_size"],
                    self.objective,
                )
                state = None
                for left in range(0, len(x), self.tbptt):
                    right = min(left + self.tbptt, len(x))
                    optimizer.zero_grad(set_to_none=True)
                    p, state = self.model(x[None, left:right], state)
                    selected = mask[left:right]
                    if selected.any():
                        target = torch.as_tensor(
                            y[left:right][selected],
                            device=self.model.device,
                            dtype=torch.float32 if self.objective == "regression" else torch.long,
                        )
                        loss = self._loss(
                            p[0, torch.tensor(selected, device=self.model.device)], target
                        )
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite training loss")
                        loss.backward()
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.grad_clip, error_if_nonfinite=True
                        )
                        optimizer.step()
                    state = state.detach()
            metric = self.evaluate(validation)
            history.append({"epoch": epoch + 1, **metric})
            if metric["loss"] < best:
                best = metric["loss"]
                # The graph is fixed: do not duplicate millions of edges each epoch.
                best_state = {
                    name: parameter.detach().clone()
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                }
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name in best_state:
                    parameter.copy_(best_state[name])
        self.model.eval()
        return {
            "objective": self.objective,
            "mode": self.model.config["mode"],
            "seed": self.seed,
            "train_ids": [e.id for e in train],
            "validation_ids": [e.id for e in validation],
            "history": history,
            "best_validation_loss": best,
            "seconds": time.perf_counter() - start,
            "trainable_parameters": sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            ),
            "graph_fingerprint": self.model.graph.fingerprint,
            "tbptt": self.tbptt,
            "optimizer": "Adam",
            "learning_rate": self.lr,
            "gradient_clip_norm": self.grad_clip,
            "model_seed": self.model.config["seed"],
            "torch_version": str(torch.__version__),
            "device": str(self.model.device),
            "scope": "fixed edges; learned adapters and optionally bounded cell-wise dynamics",
        }
