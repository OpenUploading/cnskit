from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from cns_tinker.actions.adapters import ActionOutput, build_action_adapter
from cns_tinker.evidence.export import EvidenceBundle, export_evidence_bundle
from cns_tinker.runtime.connectome import Connectome, RuntimeState
from cns_tinker.scenario.schema import Scenario, load_scenario
from cns_tinker.sensory.adapters import build_sensory_adapter, merge_drives


@dataclass(frozen=True)
class RunResult:
    scenario_id: str
    output_dir: Path
    evidence: EvidenceBundle


def run_scenario(scenario_or_path: Scenario | str | Path, output_dir: str | Path) -> RunResult:
    scenario = (
        load_scenario(scenario_or_path)
        if isinstance(scenario_or_path, (str, Path))
        else scenario_or_path
    )
    connectome = Connectome()
    sensory_adapters = [build_sensory_adapter(spec) for spec in scenario.stimuli]
    action_adapter = build_action_adapter(scenario.action)
    dt_ms = 1000.0 / scenario.world.observation_hz
    steps = max(1, int(scenario.objective.episode_seconds * scenario.world.observation_hz))
    steps = min(steps, 1200)

    state = connectome.initial_state(seed=7)
    states: list[RuntimeState] = []
    stimulus_rows: list[Mapping[str, object]] = []
    action_rows: list[ActionOutput] = []

    for step in range(steps):
        observation = _observation_for_step(scenario, step, steps)
        drives = [adapter.encode(observation) for adapter in sensory_adapters]
        drive = merge_drives(drives)
        state = connectome.step(state, drive, dt_ms=dt_ms)
        action = action_adapter.decode(state)
        states.append(state)
        stimulus_rows.append({"t_ms": state.t_ms, **observation, "drive_summary": drive.summary})
        action_rows.append(action)

    evidence = export_evidence_bundle(
        scenario=scenario,
        connectome=connectome,
        states=states,
        stimulus_rows=stimulus_rows,
        action_rows=action_rows,
        output_dir=output_dir,
    )
    return RunResult(
        scenario_id=scenario.scenario_id,
        output_dir=Path(output_dir),
        evidence=evidence,
    )


def _observation_for_step(scenario: Scenario, step: int, steps: int) -> dict[str, float]:
    phase = step / max(steps - 1, 1)
    obs: dict[str, float] = {"phase": phase}
    task = scenario.world.task
    modalities = {spec.modality for spec in scenario.stimuli}
    if "visual" in modalities:
        loom_peak = max(0.0, 1.0 - abs(phase - 0.42) / 0.16)
        light_bias = 0.5 + 0.5 * phase if task == "phototaxis" else 0.35
        obs.update({"expansion_rate": loom_peak, "contrast": light_bias, "bearing": phase - 0.5})
    if "gustatory" in modalities:
        pulse = 1.0 if 0.35 <= phase <= 0.72 else 0.0
        obs.update({"sugar": pulse, "water": 0.4 * pulse, "salt": 0.0, "bitter": 0.0})
    if "olfactory" in modalities:
        plume = max(0.0, 1.0 - abs(phase - 0.48) / 0.24)
        obs.update(
            {
                "odor_concentration": plume,
                "odor_gradient": phase - 0.48,
                "odor_hit": 1.0 if plume > 0.55 else 0.0,
            }
        )
    if "mechanosensory" in modalities:
        obs.update(
            {
                "wind_speed": max(0.0, 1.0 - abs(phase - 0.55) / 0.22),
                "contact": 0.25,
                "tilt": 0.05,
            }
        )
    if "synthetic" in modalities:
        left = 1.0 if int(phase * 8) % 2 == 0 else 0.0
        obs.update({"target_left": left, "target_right": 1.0 - left, "urgency": phase})
    return obs
