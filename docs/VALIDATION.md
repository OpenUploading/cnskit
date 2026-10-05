# Validation record

Measured 2026-10-04 (Pacific time), Windows, Intel Core i9-13900H, Python 3.13.14, PyTorch 2.14.1+cpu. CPU experiments use four Torch threads. Timings are one-machine observations, not throughput guarantees.

## Numerical and API checks

31 automated tests cover legacy graph import/runtime behavior plus sparse/dense recurrence agreement, finite-difference adapter gradients, episode isolation, training-only normalization, masked classification, fixed-mode parameters, streaming chunk equivalence, CLI training/prediction, checkpoint replay and corruption rejection. The test suite uses tiny artificial fixtures, not remote dataset downloads. The [GitHub Actions workflow](https://github.com/OpenUploading/cnskit/actions/workflows/tests.yml) runs the suite and package build on Python 3.11 and 3.13; consult the linked run for its current status.

## Real MaleCNS graph

[Machine-readable result](evidence/malecns-cpu.json). Official traced-body graph: **165,122 neurons, 25,563,197 directed edges**. Five full-graph forward steps matched independent SciPy recurrence to **5.96e-8 maximum absolute error**. Those five steps took 2.32 seconds on this machine, excluding loading and model construction. This is a numerical check, not a real-time benchmark.

An explicitly selected 128-cell / 2,291-edge induced graph trained on a synthetic temporal-filter task. Test MSE changed from 0.32795 before fitting to 0.000556 after validation-based selection; exported/reloaded predictions were bit-identical. The task has 12 training, 4 validation and 4 test episodes of 32 steps, using separate random seeds. This tests the software path with real anatomy; the targets are not fly behavior.

```sh
python examples/validate_malecns.py --graph /data/traced-graph --out runs/real-validation
```

## Does graph structure help this example?

[Machine-readable comparison](evidence/synthetic-baselines.json). Same temporal inputs, episode splits, 30 epochs, Adam learning rate 0.02, gradient norm cap 1.0, seed 7 and validation selection. A whole 32-step episode fits the truncation window. The RNN has five hidden units; parameters are close but not identical.

| Model | Trainable parameters | Test MSE |
|---|---:|---:|
| CNSKit, synthetic sparse graph | 57 | 0.001380 |
| CNSKit, zero-edge ablation | 57 | 0.000702 |
| Vanilla RNN | 51 | 0.000937 |

**The graph does not improve this simple task.** Leaky cell state alone provides temporal memory. These results demonstrate working adaptation, and show why an observation-only baseline is insufficient evidence for a connectome advantage. They are one-seed engineering examples, not a leaderboard or statistical comparison.

```sh
python examples/compare_baselines.py
python examples/train_sequence.py --out runs/sequence
```

Full-graph training, CUDA speedups, biological behavior, population-level fidelity and general reasoning are not established. Substantive task studies need matched information, parameter/compute budgets, multiple seeds and anatomical ablations.
