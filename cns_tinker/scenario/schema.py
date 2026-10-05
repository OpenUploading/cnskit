from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cns_tinker.actions.adapters import ActionSpec
from cns_tinker.api_types import FrozenMapping
from cns_tinker.sensory.adapters import SensorySpec


@dataclass(frozen=True)
class WorldSpec:
    backend: str
    task: str
    observation_hz: int = 60

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> WorldSpec:
        hz = int(payload.get("observation_hz", 60))
        if hz <= 0:
            raise ValueError("world.observation_hz must be positive")
        return cls(backend=str(payload["backend"]), task=str(payload["task"]), observation_hz=hz)


@dataclass(frozen=True)
class AdaptationSpec:
    preset: str
    trainable_params_max: int = 0
    connectome_edges: str = "frozen"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> AdaptationSpec:
        edges = str(payload.get("connectome_edges", "frozen"))
        if edges != "frozen":
            raise ValueError("adaptation.connectome_edges must be 'frozen' for the public scaffold")
        return cls(
            preset=str(payload.get("preset", "none")),
            trainable_params_max=int(payload.get("trainable_params_max", 0)),
            connectome_edges=edges,
        )


@dataclass(frozen=True)
class ObjectiveSpec:
    reward: str
    episode_seconds: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ObjectiveSpec:
        seconds = float(payload.get("episode_seconds", 10.0))
        if seconds <= 0:
            raise ValueError("objective.episode_seconds must be positive")
        return cls(reward=str(payload.get("reward", "inspectable_run")), episode_seconds=seconds)


@dataclass(frozen=True)
class EvidenceSpec:
    record_video: bool = False
    record_spike_view: bool = True
    export_run_log: bool = True

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> EvidenceSpec:
        return cls(
            record_video=bool(payload.get("record_video", False)),
            record_spike_view=bool(payload.get("record_spike_view", True)),
            export_run_log=bool(payload.get("export_run_log", True)),
        )


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    world: WorldSpec
    stimuli: tuple[SensorySpec, ...]
    action: ActionSpec
    adaptation: AdaptationSpec
    objective: ObjectiveSpec
    evidence: EvidenceSpec
    metadata: FrozenMapping

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> Scenario:
        stimuli_payload = payload.get("stimuli", {})
        if not isinstance(stimuli_payload, Mapping):
            raise ValueError("stimuli must be a mapping")
        stimuli = tuple(
            SensorySpec.from_mapping(str(modality), spec)
            for modality, spec in stimuli_payload.items()
        )
        if not stimuli:
            raise ValueError("scenario must define at least one stimulus adapter")
        return cls(
            scenario_id=str(payload["scenario_id"]),
            world=WorldSpec.from_mapping(_mapping(payload["world"], "world")),
            stimuli=stimuli,
            action=ActionSpec.from_mapping(_mapping(payload["actions"], "actions")),
            adaptation=AdaptationSpec.from_mapping(
                _mapping(payload.get("adaptation", {}), "adaptation")
            ),
            objective=ObjectiveSpec.from_mapping(
                _mapping(payload.get("objective", {}), "objective")
            ),
            evidence=EvidenceSpec.from_mapping(_mapping(payload.get("evidence", {}), "evidence")),
            metadata=FrozenMapping(dict(_mapping(payload.get("metadata", {}), "metadata"))),
        )

    @classmethod
    def from_parts(
        cls,
        *,
        scenario_id: str,
        world: Any,
        sensory: tuple[SensorySpec, ...],
        action: ActionSpec,
        adaptation_preset: str,
        evidence_enabled: bool,
    ) -> Scenario:
        return cls(
            scenario_id=scenario_id,
            world=WorldSpec(
                backend=str(world.backend),
                task=str(world.task),
                observation_hz=int(world.observation_hz),
            ),
            stimuli=sensory,
            action=action,
            adaptation=AdaptationSpec(preset=adaptation_preset, trainable_params_max=0),
            objective=ObjectiveSpec(reward="sdk_run", episode_seconds=10.0),
            evidence=EvidenceSpec(
                record_video=False,
                record_spike_view=evidence_enabled,
                export_run_log=True,
            ),
            metadata=FrozenMapping({"source": "sdk"}),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "world": {
                "backend": self.world.backend,
                "task": self.world.task,
                "observation_hz": self.world.observation_hz,
            },
            "stimuli": {
                spec.modality: {
                    "mapper": spec.mapper,
                    "realism_level": spec.realism_level,
                    "target_regions": list(spec.target_regions),
                }
                for spec in self.stimuli
            },
            "actions": {
                "decoder": self.action.decoder,
                "target": self.action.target,
                "action_space": list(self.action.action_space),
            },
            "adaptation": {
                "preset": self.adaptation.preset,
                "trainable_params_max": self.adaptation.trainable_params_max,
                "connectome_edges": self.adaptation.connectome_edges,
            },
            "objective": {
                "reward": self.objective.reward,
                "episode_seconds": self.objective.episode_seconds,
            },
            "evidence": {
                "record_video": self.evidence.record_video,
                "record_spike_view": self.evidence.record_spike_view,
                "export_run_log": self.evidence.export_run_log,
            },
            "metadata": dict(self.metadata),
        }


def load_scenario(path: str | Path) -> Scenario:
    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as handle:
        if scenario_path.suffix.lower() == ".json":
            import json

            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"scenario file did not contain a mapping: {scenario_path}")
    return Scenario.from_mapping(payload)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
