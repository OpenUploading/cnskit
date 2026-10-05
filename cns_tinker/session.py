"""Local observation-to-action sessions using the synthetic preview runtime."""
from collections.abc import Mapping
import math

from cns_tinker.actions.adapters import ActionOutput, build_action_adapter
from cns_tinker.runtime.connectome import Connectome, RuntimeState
from cns_tinker.scenario.schema import Scenario
from cns_tinker.sensory.adapters import build_sensory_adapter, merge_drives


class Session:
    """Feed your own observations; no server, training or real MaleCNS required.

    One instance belongs to one environment episode. Not thread-safe.
    State snapshots are copies, so callers cannot mutate the running state.
    """

    def __init__(self, scenario: Scenario, *, seed: int = 7):
        self.runtime = Connectome(seed=seed)
        self.adapters = [build_sensory_adapter(s) for s in scenario.stimuli]
        self.decoder = build_action_adapter(scenario.action)
        hz = float(scenario.world.observation_hz)
        if not math.isfinite(hz) or hz < 25:
            raise ValueError("Session observation_hz must be at least 25")
        self.dt_ms = 1000.0 / hz
        self._state = self.runtime.initial_state()

    @property
    def state(self) -> RuntimeState:
        return RuntimeState(self._state.t_ms, self._state.activity.copy())

    def reset(self) -> RuntimeState:
        """Restart the episode with the constructor seed."""
        self._state = self.runtime.initial_state()
        return self.state

    def step(self, observation: Mapping[str, float]) -> ActionOutput:
        """Advance one fixed timestep from application-owned observations."""
        clean = {key: float(value) for key, value in observation.items()}
        if not all(isinstance(k, str) and math.isfinite(v) for k, v in clean.items()):
            raise ValueError("Observation keys must be strings and values finite numbers")
        drive = merge_drives([adapter.encode(clean) for adapter in self.adapters])
        self._state = self.runtime.step(self._state, drive, dt_ms=self.dt_ms)
        return self.decoder.decode(self._state)
