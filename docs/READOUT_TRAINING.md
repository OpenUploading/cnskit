# Train a task readout

CNSKit can now fit a **ridge regression readout** locally using NumPy. The connectome and its dynamics remain frozen. There is no end-to-end graph fine-tuning, automatic policy training or task-language interpreter.

```sh
python examples/train_readout.py --out runs/my-readout.json
```

Create `runs/` first if absent. Use a new checkpoint filename. The example runs four synthetic training episodes and one separately seeded test episode, prints held-out MSE and a training-mean baseline, and saves a JSON checkpoint. Labels are authored steering targets, not measured animal behavior. This is not a real MaleCNS performance result.

```python
from cns_tinker.readout import RidgeReadout

model = RidgeReadout.fit(train_activity, train_targets,
    feature_names=body_id_names, output_names=["steer"], alpha=1.0)
metrics = model.evaluate(test_activity, test_targets, feature_names=body_id_names)
model.save("my-readout.json")
loaded = RidgeReadout.load("my-readout.json")
actions = loaded.predict(current_activity, feature_names=body_id_names)
```

Inputs and targets are finite 2D arrays: samples x selected channels, samples x outputs. Feature names must match exactly, including order, when predicting. For real MaleCNS use explicit body-ID strings and selected-state traces; never assume array positions have the same meaning between graphs. Keep graph hash and task provenance with your experiment; the current checkpoint validates channel identity/order but does not bind a graph hash automatically.

Normalization is fitted only on training data. Alpha must be positive; select it on a validation set, never the final test set. Split by whole episode/animal/session as appropriate before fitting; random neighboring-frame splits can leak temporal information. The API cannot enforce your split, label quality or biological interpretation.

The runtime chooses the smaller primal/dual dense linear solve. Use a bounded readout population, not all 165k neurons at once. Outputs are continuous, unconstrained values, not probabilities or safety-limited controls. Applications own clamping, action decoding and evaluation. JSON loading uses no pickle; checkpoint saving refuses overwrite.
