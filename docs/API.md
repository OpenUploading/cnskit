# Python and CLI reference

[Documentation](README.md) / API

The names below are source APIs, not a stable 1.0 compatibility promise. Import namespace: `cns_tinker`.

## Synthetic Session

```python
from cns_tinker.session import Session
from cns_tinker.scenario.schema import load_scenario
session = Session(load_scenario("recipes/01_photon_saber_readout/scenario.yaml"), seed=7)
```

| Member | Contract |
| --- | --- |
| `Session(scenario, *, seed=7)` | Existing adapters; synthetic 512-unit runtime; observation rate must be finite and >=25 Hz |
| `step(observation)` | String-keyed finite numeric mapping; advances one fixed timestep; returns `ActionOutput` |
| `state` | Snapshot containing `t_ms` and a copied activity array |
| `reset()` | Reset state using constructor seed; returns a state snapshot |

`ActionOutput` exposes `action`, `values`, `confidence`. Read decoder source for the interpretation of values; confidence is an authored score. The session does not export history automatically.

## Real MaleCNSSession

```python
from cns_tinker.runtime.malecns import MaleCNSSession
with MaleCNSSession("/path/to/traced-graph", dt_ms=5, gain=0.8, workers=1) as session:
    inputs = session.select(cell_type="LC4")
    outputs = session.select(cell_type="DNp01")
    session.step(dict.fromkeys(inputs, 1.0))
    activity = session.read(outputs)
```

| Member | Contract |
| --- | --- |
| Constructor | Verified prepared graph; `0 < dt_ms <= 20`; `0 <= gain < 1`; integer workers 1–32 |
| `select(*, cell_type=None, superclass=None)` | At least one filter; exact official annotation labels; combined with AND; list of body IDs |
| `step(currents)` | Integer body IDs to finite current values, magnitude <=100 a.u.; advances `dt_ms`; returns copied state |
| `read(body_ids)` | Dictionary of requested ID to current activity in a.u.; unknown/excluded IDs raise |
| `state` | Full copied state in the order of `body_ids`; copying the full graph has a cost |
| `reset()` | Zero state and clock; returns `None` |
| `record(inputs, *, readout_ids, out, max_steps=10000)` | Reset session; 1–10,000 steps; nonempty unique readout IDs; fresh output directory; returns output Path |
| `truth_metadata()` | Graph source, dynamics and numerical settings |
| `close()` | Release worker pool; subsequent stepping/recording fails |

Use a context manager for workers. Sessions are not thread-safe. `record()` computes the whole selected graph but exports only selected readout channels. Invalid later frames can advance the state before raising; reset before retrying. There is no silent fallback to a synthetic graph.

## Local RidgeReadout

Import `RidgeReadout` from `cns_tinker.readout`. This fits a continuous-output
mapping from selected activity channels while leaving the runtime graph fixed.

| Member | Contract |
| --- | --- |
| `fit(activity, targets, *, feature_names, output_names, alpha=1.0)` | Finite 2D arrays with aligned rows, at least two samples, unique ordered channel names, positive ridge penalty; returns a fitted readout |
| `predict(activity, *, feature_names)` | Uses training-set scaling; requires exactly the saved feature order; returns a sample-by-output array |
| `evaluate(activity, targets, *, feature_names)` | Mean squared error for each output; use held-out episodes |
| `save(path)` | Writes a new JSON checkpoint; refuses to overwrite an existing file |
| `load(path)` | Validates the checkpoint schema, array dimensions and numeric values |

The checkpoint validates feature names, not the identity of the upstream graph.
Keep graph provenance alongside it. This is a dense CPU solver for selected
channels, not full-connectome weight training. See [Readout training](READOUT_TRAINING.md)
for the runnable synthetic example and split protocol.

## CLI

```sh
python -m cns_tinker.cli validate SCENARIO
python -m cns_tinker.cli run SCENARIO --out runs/new-example
python -m cns_tinker.runtime.malecns --data-dir RAW --out PREPARED
python -m cns_tinker.cli real-run --graph PREPARED --inputs currents.json --readout-type DNp01 --workers 1 --out runs/new-real-run
```

`currents.json` is a JSON array of 1–10,000 objects. Keys are canonical integer body IDs encoded as strings; values are numeric currents. File limit: 32 MiB. An empty object means zero external drive for that timestep. Default model timestep is 5 ms for `real-run`.

```json
[{"12032": 1.0}, {"12032": 1.0}, {}]
```

The ID must exist in your prepared graph. It is an explicit injection, not a sensory transduction model. Run `--help` on each command for current options.
