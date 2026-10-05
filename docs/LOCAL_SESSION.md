# Local application integration

CNSKit is a Python library with an optional local web viewer. The web project's npm
dependencies do not contain an inference SDK. No hosted account is required.

From a repository checkout: `python -m pip install -e .`.

```python
from cns_tinker.scenario.schema import load_scenario
from cns_tinker.session import Session

scenario = load_scenario('recipes/01_photon_saber_readout/scenario.yaml')
session = Session(scenario, seed=7)
for _ in range(60):
    action = session.step({'target_left': 1.0, 'target_right': 0.0, 'urgency': 1.0})
    # Your application consumes action.action and action.values here.
print(action.action, session.state.t_ms)
session.reset()
```

The application owns observations and consumes outputs. `step` advances one
fixed sample, `state` returns a copy, and `reset` reuses the constructor seed.
Sessions retain only current state; use the existing recipe runner for exported
traces. Session trace export and user-supplied adapter registration remain future work.

This implementation uses the synthetic preview runtime and existing adapters.
It does not load anatomical MaleCNS data, train weights, connect a game engine,
or execute financial orders. The application must implement its own environment.
Inputs use the selected adapter's feature names; unsupported names are ignored
by existing adapters. Use observation rates of at least 25 Hz.

For explicit body-ID input on the real graph, use `MaleCNSSession` as described in [MALECNS.md](MALECNS.md). The Session class above intentionally retains the synthetic recipe backend.
