# Training a custom task

## Data and splits

Each `Episode(id, observations, targets, mask=None)` contains observations `[time,input]` and regression targets `[time,output]` or integer classification targets `[time]`. A boolean mask selects supervised timesteps; unlabelled timesteps still advance neural state. Use separate animals, sessions or environments when defining train/validation/test episodes. Adjacent frames are not independent examples.

The trainer rejects overlapping train/validation IDs. It cannot detect duplicated content assigned different IDs: callers own the semantic split. Normalization is fitted on all observations in training episodes only. Validation selects the best epoch. Call `evaluate(test_episodes)` once after model selection; do not tune against the test set.

## Training modes

- `readout`: fixed graph, fixed seeded input projection, train output projection only.
- `adapters`: train input and output projections through the recurrent dynamics; graph unchanged.
- `dynamics`: also train per-neuron gain and leak, each sigmoid-bounded to `(0,1)`. Topology and synaptic weights stay fixed. This is an adapted numerical model.

Raw synapse counts remain intact in the imported dataset. The task operator uses the parent's incoming-weight normalization. An induced subgraph drops edges to omitted cells and **does not renormalize** retained rows; it is not the full network.

For observation `x`, state `a`, sparse operator `W`, selected input injection `S`, and selected readout `R`:

```text
u = S · encoder((x - training_mean) / training_scale)
a_next = a + sigmoid(leak) * (tanh(sigmoid(gain) * W a + u) - a)
y = decoder(R a_next)
```

The default leak is `5/80`, consistent with the legacy 5 ms rate-model step. Learned leak does not identify a biological time constant. No neurotransmitter signs, spiking units or conduction delays are inferred by this operator.

## Sequence and memory semantics

Every episode starts from zero state. Training processes one episode at a time; each `tbptt` window backpropagates independently and detaches its final state before the next window. Longer dependencies than the window do not have full gradient credit assignment. At inference, a batch supports independent parallel states. Returned state is a Torch tensor on the model device; predictions are NumPy arrays.

Edges are stored sparsely and the sparse operator is constructed once. The recurrent activation tape still scales with neurons × batch × truncation length. There is no automatic full-graph memory estimator, distributed trainer, mixed precision implementation or guaranteed real-time throughput. Start with a measured subgraph run before scaling.

## Command line

### Your own training loop

`ConnectomeModel` is a regular `torch.nn.Module`. For task losses outside the built-in MSE / cross-entropy trainer, call its differentiable `forward` or `step` in your own optimization loop:

```python
import torch

optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=0.001)
optimizer.zero_grad()
predictions, state = model(observations)  # [batch, time, input_size]
loss = your_task_loss(predictions, targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

The application owns normalization, episode boundaries, loss masking and validation in a custom loop. Detach carried state at truncation boundaries. This extension point does not implement an RL algorithm or automatically make an environment differentiable.

## Dataset files and CLI

For partially labeled sequences, masked-out regression targets may be `NaN` and masked-out classification targets may be `-1`. Observations must remain finite at every timestep: unlabeled observations still advance the recurrent state. Every episode needs at least one labeled step.

Use `trainer.fit(train, validation, epochs=100, patience=5, min_delta=1e-4)` to stop after five epochs without a sufficient validation-loss improvement. `min_delta` is an absolute loss threshold measured against the last significant improvement. The model always restores the absolute lowest-loss epoch, including improvements smaller than this threshold. The report records `best_epoch`, `epochs_completed` and `stopped_early`. The default `patience=None` runs the full budget. Keep test episodes separate from validation used for stopping.

Save each split as an NPZ with `observations [episodes,time,input]`, aligned `targets`, string `episode_ids`, and optional boolean `mask [episodes,time]`. Never use object arrays. Variable-length episodes can use the Python API; a mask excludes loss, it does not remove padded transitions.

`task.json`:

```json
{
  "graph_dir": "/data/traced-graph",
  "subgraph_ids": [12032, 10001, 10010],
  "model": {"input_ids": [12032], "readout_ids": [10001, 10010], "input_size": 4, "output_size": 2, "mode": "adapters", "seed": 7},
  "trainer": {"objective": "regression", "lr": 0.01, "tbptt": 32, "seed": 7},
  "epochs": 30,
  "patience": 5,
  "min_delta": 0.0001
}
```

The three-cell selection above illustrates the file format; choose a task-relevant circuit with intermediate pathways for a meaningful experiment.

```sh
cnskit train --config task.json --train train.npz --validation validation.npz --out runs/task
cnskit predict --policy runs/task --input observations.npy --out predictions.npy
cnskit evaluate --policy runs/task --data test.npz --objective regression
```

Outputs must be new paths. `model.save()` exports an inference bundle, not an exact optimizer/RNG resume snapshot. Continue fitting with `Trainer(load_policy(...))` to start a new optimizer. Checkpoint hashes catch corruption and graph mismatches, not malicious replacement of an entire unsigned bundle. Load only trusted artifacts and keep your own provenance record.

## Evaluate whether the graph helps

### Long-sequence evaluation

`trainer.evaluate(episodes, chunk_size=1024)` processes each episode in contiguous chunks. Unlabeled chunks still advance state; separate episodes reset it. Loss is weighted by labeled timesteps, including short final chunks. Classification accuracy aggregates correct predictions across all labeled steps. Training-time validation uses the same default chunking.

```sh
cnskit evaluate --policy runs/task --data test.npz --objective regression --chunk-size 256
```

Use a positive chunk size, or `chunk_size=None` in Python for whole-episode evaluation. Chunking bounds model input and prediction buffers on the compute device, not the resident graph or dataset. The NPZ loader still loads arrays into host memory; this is not disk-streaming data loading. Loss reductions can differ slightly from whole-episode evaluation due to floating-point summation order. Chunk size does not change recurrent state transitions.

### Task comparisons

Use the same observation information, episode splits and optimization budget for an observation-only model, a recurrent baseline and graph ablations. Report multiple seeds, task error/accuracy, latency, memory, parameter count and graph selection. The included example compares with memoryless ridge; this alone does not establish an anatomical advantage. Reward-only RL and arbitrary topology training are not implemented.
