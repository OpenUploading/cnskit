# Real MaleCNS graph backend

CNSKit now has an opt-in local MaleCNS v1.0 backend. The existing `Session` and
browser game recipes remain synthetic. Use `MaleCNSSession` explicitly.

## Install and prepare

```sh
python -m pip install -e ".[malecns]"
```

Download these original files from https://male-cns.janelia.org/download/ into
a local data directory (outside the repository):

- `body-annotations-male-cns-v1.0-minconf-0.5.feather`
- `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

```sh
python -m cns_tinker.runtime.malecns --data-dir /path/to/raw --out /path/to/traced-graph
```

The importer selects `status == Traced`, keeps edges with both endpoints in that
selection, and aggregates parallel rows by summing counts. It excludes glia,
orphans and other untraced fragments. This is an induced graph, not every segment
in the full downloadable graph. Matrix orientation is `W[post, pre]`.

Verified local import: 165,122 bodies, 25,563,197 directed edges and 124,025,046
summed synapse counts. Body IDs, raw counts, annotations, source URLs, SHA-256
checksums and selection rules are retained. Graph artifacts are checked on load.

## Use your own input and readout

```python
from cns_tinker.runtime.malecns import MaleCNSSession

session = MaleCNSSession('/path/to/traced-graph', dt_ms=5)
for _ in range(100):
    state = session.step({12032: 1.0})  # explicit current in LC4 body 12032
print(session.read([10001, 10010]))    # DNp01 body IDs; raw state, not behavior
session.reset()
```

Invalid or excluded IDs fail; the backend never substitutes a synthetic graph.
Applications own the mapping from observations to these currents and from states
to actions. Arbitrary body IDs do not constitute validated sensory encodings.

### Select cells and record custom inputs

```python
with MaleCNSSession('/path/to/traced-graph', workers=4) as session:
    input_ids = session.select(cell_type='LC4')
    readout_ids = session.select(cell_type='DNp01')
    observations = [dict.fromkeys(input_ids, 1.0) if 20 <= i < 100 else {}
                    for i in range(400)]
    session.record(observations, readout_ids=readout_ids, out='runs/my-input')
```

`select` matches exact official `type` and/or `superclass` labels; combined
filters use AND. It does not infer sensory function. `record` requires a reset
session, a fresh output directory and 1–10,000 input frames. It stores actual
currents with interval timestamps, selected activity, body IDs, graph provenance,
numerical settings and output SHA-256 checksums. Missing currents mean zero.
Invalid later inputs may advance the session before raising; reset before retry.

For a JSON array of per-step body-ID/current objects, the CLI runs the same API:

```sh
cns-tinker real-run --graph /path/to/traced-graph --inputs currents.json --readout-type DNp01 --workers 4 --out runs/custom
```

Example JSON shape: `[{"12032": 1.0}, {"12032": 1.0}, {}]`. Each item advances
5 ms under default settings. This is explicit current injection, not an arbitrary
environment description or a trained task adapter.

### CPU parallelism

`workers=1` remains the default. Optional row-partitioned CPU workers preserve
each row's accumulation order. Use a context manager or call `close()` to release
worker threads. A session is not safe for concurrent calls. Partition copies use
additional sparse-matrix memory and workers compete with other apps such as UE.

Local paired check on 2026-09-23, same full graph and 400-step LC4 sequence:
one worker 9.07 s, eight workers 4.71 s, including record export but excluding
graph loading. Selected DNp01 traces matched exactly. A separate full-state
fixture test checks serial/parallel equality. These are single-machine timing
measurements; 2 s model time still takes longer than real time. No GPU claim.

## Reproduce the probe

```sh
python scripts/run_malecns_probe.py --graph /path/to/traced-graph --out runs/real-probe
```

The script injects 1 a.u. into annotated LC4 cells from 100 through 500 ms and
records two DNp01 states plus up to 62 LC4 channels. The whole selected graph is
computed; only the explicitly listed display channels are exported. Outputs:
`manifest.json`, `stimulus_trace.csv`, `neural_trace.npz`. A fresh output directory
is required. The manifest names every injected/read body ID and numerical setting.

Local run `malecns_v1_probe_20260922`: 400 steps, 2,000 ms model time, 13.57 s
stepping wall time (excludes load; one measurement, not a throughput benchmark).
DNp01 peaks: body 10001 = 0.108758 a.u.; body 10010 = 0.121881 a.u.
Pre-input sampled activity is zero. Readout IDs are not injection IDs. With
recurrence gain zero, the DNp01 readouts stay zero under the same injection.
This tests implementation dependence on recurrence, not biological validity.

## Model boundary

Raw anatomy is real. Dynamics are an authored unsigned leaky-rate approximation:
incoming weights normalized by each receiving cell's total count; gain 0.8;
tau 80 ms; tanh nonlinearity; fixed 5 ms steps; zero initial state. These choices
are not estimated from the animal. Neurotransmitter signs, conduction delays,
calibrated sensory transduction and trained behavioral readouts are absent.
This backend reports rates in arbitrary units, not measured spikes.

The website's MaleCNS panel plays a saved local probe, not live browser inference.
Its original game runner remains synthetic and is labeled separately. Loading a
connectome does not automatically make its outputs a game-playing controller.

## Attribution

MaleCNS v1.0: FlyEM (HHMI Janelia), University of Cambridge, MRC Laboratory of
Molecular Biology and Google Research. Source: https://male-cns.janelia.org/download/
License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
Modifications: traced-body selection, sparse conversion, incoming normalization
for synthetic dynamics and derived sampled activity. No endorsement is implied.
