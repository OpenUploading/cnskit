# CNSKit

[![SDK checks](https://github.com/OpenUploading/cnskit/actions/workflows/tests.yml/badge.svg)](https://github.com/OpenUploading/cnskit/actions/workflows/tests.yml)

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache_2.0-427B80)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-6872AD)](pyproject.toml)

**Train task-specific models on a connectome. Run them in your own application.**

CNSKit connects MaleCNS graph data to a practical Python workflow: bind observations to selected neurons, train a readout or input adapter, evaluate held-out episodes, and export a stateful inference model. The graph stays sparse and its body IDs and source provenance travel with the model.

[Quickstart](#quickstart)  |  [Training guide](docs/TRAINING.md)  |  [MaleCNS data](docs/MALECNS.md)  |  [API](docs/LEARNING_API.md)  |  [Evidence](docs/VALIDATION.md)

![CNSKit: graph-grounded task training and stateful inference](docs/assets/cnskit-overview.svg)

## What you can do

| Workflow | Implementation |
|---|---|
| Custom sequence regression or classification | Episode datasets, partial labels, early stopping, Adam and truncated backpropagation |
| Fixed-graph experimentation | Train only a readout, input/output adapters, or adapters plus bounded per-neuron gains and leak |
| MaleCNS integration | Verified graph import, exact body-ID binding, explicit induced subgraphs |
| Stateful inference | Batched sequences or one step at a time; state is passed explicitly |
| Reproducible export | NumPy checkpoints without pickle, graph fingerprint, ordered channels, fitted normalization and training report |
| Lightweight baseline | NumPy ridge readout fitting remains available without PyTorch |

The framework is task-general; it is not a pretrained general intelligence model. Anatomical connectivity is an experimental prior. Dynamics, observation encodings and learned outputs are model choices. We do not claim biological fidelity, a MaleCNS advantage over ordinary networks, or whole-connectome training speedups.

## Quickstart

Python 3.11 or newer. No account, server or connectome download is needed for the synthetic example.

```sh
git clone https://github.com/OpenUploading/cnskit.git
cd cnskit
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1
python -m pip install -e ".[train,malecns]"
python examples/train_sequence.py --out runs/first-policy
```

This fits a temporal filtering task on a **synthetic graph**, evaluates separate test episodes, compares against observation-only ridge regression, and verifies checkpoint reload. It demonstrates the workflow, not MaleCNS performance. PyTorch is optional; install the appropriate CPU/CUDA wheel for your environment before the package if needed.

For masked sequence classification, run `python examples/train_classification.py --out runs/classifier`. For matched recurrent and zero-edge baselines, run `python examples/compare_baselines.py`.

```python
from cnskit import load_policy
import numpy as np

policy = load_policy("runs/first-policy")
prediction, state = policy.predict(np.zeros((1, 16, 2), dtype="float32"))
# Continue the SAME episode by passing state; omit it to start a new episode.
prediction, state = policy.predict(np.ones((1, 8, 2), dtype="float32"), state=state)
```

## Run a complete MaleCNS experiment

Prepare the official data using the [dataset guide](docs/MALECNS.md), then run:

```sh
python examples/malecns_task.py --graph /data/traced-graph --out runs/malecns-task
```

This is **real MaleCNS anatomy with a synthetic regression task**. The script selects 128 neurons deterministically from the loaded graph, fits input/output adapters, and evaluates three initialization seeds against a zero-edge ablation using the same observations, splits and training budget. It exports the first requested seed, never the seed with the best test score.

| Output | What it contains |
|---|---|
| `task.json` | Actual selected body IDs and a reusable CLI configuration |
| `train.npz`, `validation.npz`, `test.npz` | Separate episodes with inputs, targets and IDs |
| `policy/` | Graph-bound model, fitted normalization and training report |
| `results.json` | Per-seed held-out errors and the ablation comparison |
| `observations.npy`, `predictions.npy` | A replay input and verified inference output |

Evaluate or use the exported model without rerunning training:

```sh
cnskit evaluate --policy runs/malecns-task/policy --data runs/malecns-task/test.npz --objective regression
cnskit predict --policy runs/malecns-task/policy --input runs/malecns-task/observations.npy --out runs/replayed.npy
```

The example checks exact checkpoint replay and chunked stateful inference. Evaluation processes at most 1,024 timesteps per model call by default; use `--chunk-size 256` for smaller prediction buffers. Dataset arrays and the graph still reside in memory. Output paths must be new. It loads the full prepared graph before selecting a subgraph; selection does not eliminate the import's memory requirement.

## Bring your own task

Supply observations `[episode, time, input]`, regression targets `[episode, time, output]`, unique string `episode_ids`, and an optional boolean `mask` in each NPZ split. Masked labels may be missing; all observations must remain finite. Split by independent sessions or environments rather than adjacent frames.

```sh
python examples/malecns_task.py --graph /data/traced-graph --out runs/my-task --train train.npz --validation validation.npz --test test.npz
```

Input/output dimensions are inferred from your data. The default neuron selection is an engineering starting point, not a validated sensory-to-motor pathway. Use the exported `task.json` and [Python API](docs/LEARNING_API.md) to define a task-relevant circuit, named channels and adaptation mode. For classification, see the [classification example](examples/train_classification.py) and [training guide](docs/TRAINING.md).

## Evidence and current limits

- **Numerical correctness:** five full-graph forward steps on 165,122 neurons and 25,563,197 edges matched an independent SciPy calculation to 5.96e-8 maximum absolute error.
- **Software verification:** 43 tests cover training, masking, early stopping, explicit-state inference, CLI evaluation and checkpoint integrity; CI runs Python 3.11 and 3.13.
- **Task value:** the included temporal-filter experiments test the workflow. They do not establish a MaleCNS advantage; the published synthetic-graph comparison favors the zero-edge ablation.

See [measurements and reproducible commands](docs/VALIDATION.md). Full-graph differentiation, CUDA throughput, optimizer-state resume and biological behavior validation remain open work. The SDK is suitable for controlled research experiments; it is not yet a production training platform.

## Compatibility and scope

`cnskit` is the training API and CLI. The distribution name `cns-tinker` and existing `cns_tinker` imports remain for compatibility. This is a source release, not a claim of a published PyPI package. Legacy scenario APIs and synthetic game recipes remain available; their configuration-only cloud job interfaces are separate from the new local trainer.

Apache-2.0. Data is downloaded separately under upstream terms. See [licensing](docs/LICENSING.md).

## Contribute

Bring a task, a reproducible baseline, or an execution improvement. See [contributing](CONTRIBUTING.md), [validation](docs/VALIDATION.md) and the [roadmap](docs/ROADMAP.md). The most useful next result is a well-controlled task study that establishes when anatomical structure helps.
