# Task API

| Entry point | Contract |
|---|---|
| `Graph.from_malecns(path, body_ids=None)` | Verify prepared MaleCNS artifacts; optional explicit induced subgraph |
| `Graph.synthetic(n, density, seed)` | Labelled synthetic sparse fixture for development |
| `Graph.indices(ids)` | Validate exact ordered body IDs; fail on unknown or duplicated IDs |
| `ConnectomeModel(graph, input_ids=..., readout_ids=..., input_size=..., output_size=..., mode=...)` | Build fixed-topology rate model with task adapters |
| `model.step(x, state=None)` | Torch output and next state for `[batch,input]`; differentiable |
| `model(x, state=None)` | Torch sequence output for `[batch,time,input]`; differentiable |
| `model.predict(x, state=None)` | No-gradient NumPy predictions plus explicit detached Torch state |
| `Trainer(model, objective, lr, tbptt, grad_clip, seed)` | Local supervised sequence trainer |
| `trainer.fit(train, validation, epochs=30, patience=None, min_delta=0.0)` | Fit normalization, optionally stop early, restore best validation epoch, return report |
| `trainer.evaluate(episodes)` | Reset per episode; masked MSE or cross-entropy, plus classification accuracy |
| `model.save(new_directory, report=...)` | Save graph, parameters and provenance without pickle |
| `load_policy(path, device='cpu', expected_graph=None)` | Verify file hashes and graph/channel binding before loading |

Install the `[train]` extra for the learning API. Graph utilities use NumPy/SciPy only. No implicit network download or cloud execution occurs. Models do not keep a hidden per-client state; the application must retain and pass each client's state separately. A model or trainer should not be mutated concurrently with inference.

`input_names` and `output_names` are optional ordered unique strings; defaults are `x0...` and `y0...`. They are saved in `model.config` and the manifest. Arrays must follow that order; the API does not infer units or reorder columns. Classification outputs are logits; apply softmax for probabilities or argmax for a class index.

The returned recurrent state has shape `[batch, number_of_neurons]`, float32, on the model device. Reset with `None` between independent episodes. Use `.to('cuda')` for a compatible PyTorch installation; CUDA performance is not characterized in this release. `save()` preserves an inference bundle; optimizer state is not exported.
