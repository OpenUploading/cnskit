# Installation and first run

[Documentation](README.md) / Getting started

## Requirements

Python >=3.11 and Git. A GPU, GCP account and Unreal installation are not required. Start with the [training quickstart](../README.md#quickstart) for the new `cnskit` API. The commands below describe the retained legacy scenario runner. Full MaleCNS needs additional RAM and disk; begin with the synthetic example.

Clone `https://github.com/OpenUploading/cnskit.git` and work from its root. Create `.venv` with `python -m venv .venv`. Activate with `.\.venv\Scripts\Activate.ps1` on PowerShell or `source .venv/bin/activate` on macOS/Linux.

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python`; changing execution policy is unnecessary.

```sh
python -m pip install -e .
python -m cns_tinker.cli --help
python -m cns_tinker.cli validate recipes/01_photon_saber_readout/scenario.yaml
python -m cns_tinker.cli run recipes/01_photon_saber_readout/scenario.yaml --out runs/first-saber
python examples/local_task.py
```

Expected CLI messages begin with `valid scenario:` and `wrote evidence bundle:`. The application example prints actions and model time. This is simulated state, not a biological measurement or trained skill.

## Inspect the run

```python
import json
import numpy as np
from pathlib import Path

root = Path("runs/first-saber")
manifest = json.loads((root / "manifest.json").read_text())
with np.load(root / "neural_trace.npz") as trace:
    print(manifest["steps"], trace["activity"].shape)
    print(trace["t_ms"][0], trace["t_ms"][-1])
```

| File | Meaning |
| --- | --- |
| `manifest.json` | Scenario, runtime identity and export inventory |
| `scenario.yaml` | Resolved configuration |
| `stimulus_trace.csv` | Input features at model timestamps, in milliseconds |
| `action_trace.csv` | Authored decoder labels and confidence; not calibrated probability |
| `neural_trace.npz` | `activity` (sample x unit, a.u.) and `t_ms` |
| `metrics.csv` | Runner diagnostics; not task-success validation |
| `truth_ledger.md` | Data/model claim boundary |

The recipe runner synthesizes observations and caps execution at 1,200 steps. A `world.backend: unreal` field does not launch or connect UE. Use `Session.step()` for application-owned observations. Synthetic run export can overwrite files in an existing directory; use a fresh path.

## Optional dependencies

```sh
python -m pip install -e ".[malecns]"       # real graph import and CPU runtime
python -m pip install -e ".[server-test]"   # local service and Python tests
```

Continue with [real MaleCNS](MALECNS.md) or [custom tasks](CUSTOM_TASKS.md).
