from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from cns_tinker.actions.adapters import ActionOutput
from cns_tinker.runtime.connectome import Connectome, RuntimeState
from cns_tinker.scenario.schema import Scenario


@dataclass(frozen=True)
class EvidenceBundle:
    output_dir: Path
    manifest_path: Path
    truth_ledger_path: Path


def export_evidence_bundle(
    *,
    scenario: Scenario,
    connectome: Connectome,
    states: Sequence[RuntimeState],
    stimulus_rows: Sequence[Mapping[str, object]],
    action_rows: Sequence[ActionOutput],
    output_dir: str | Path,
) -> EvidenceBundle:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario_path = out / "scenario.yaml"
    with scenario_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(scenario.to_mapping(), handle, sort_keys=False)

    _write_stimulus_trace(out / "stimulus_trace.csv", stimulus_rows)
    _write_action_trace(out / "action_trace.csv", states, action_rows)
    _write_metrics(out / "metrics.csv", states, action_rows)

    neural = np.stack([state.activity for state in states], axis=0)
    np.savez_compressed(
        out / "neural_trace.npz",
        activity=neural,
        t_ms=np.array([s.t_ms for s in states]),
    )

    manifest = {
        "scenario_id": scenario.scenario_id,
        "world": scenario.world.__dict__,
        "action_decoder": scenario.action.__dict__,
        "adaptation": scenario.adaptation.__dict__,
        "steps": len(states),
        "duration_ms": states[-1].t_ms if states else 0.0,
        "connectome": connectome.truth_metadata(),
        "artifacts": [
            "scenario.yaml",
            "truth_ledger.md",
            "stimulus_trace.csv",
            "action_trace.csv",
            "metrics.csv",
            "neural_trace.npz",
        ],
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    truth_ledger_path = out / "truth_ledger.md"
    truth_ledger_path.write_text(_truth_ledger(scenario, connectome), encoding="utf-8")
    return EvidenceBundle(
        output_dir=out,
        manifest_path=manifest_path,
        truth_ledger_path=truth_ledger_path,
    )


def _write_stimulus_trace(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_action_trace(
    path: Path,
    states: Sequence[RuntimeState],
    actions: Sequence[ActionOutput],
) -> None:
    fields = ["t_ms", "action", "confidence"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for state, action in zip(states, actions, strict=True):
            writer.writerow(
                {
                    "t_ms": state.t_ms,
                    "action": action.action,
                    "confidence": action.confidence,
                }
            )


def _write_metrics(
    path: Path,
    states: Sequence[RuntimeState],
    actions: Sequence[ActionOutput],
) -> None:
    mean_abs = (
        float(np.mean([np.mean(np.abs(state.activity)) for state in states]))
        if states
        else 0.0
    )
    mean_conf = float(np.mean([action.confidence for action in actions])) if actions else 0.0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "unit"])
        writer.writeheader()
        writer.writerow({"metric": "mean_abs_activity", "value": mean_abs, "unit": "a.u."})
        writer.writerow(
            {
                "metric": "mean_action_confidence",
                "value": mean_conf,
                "unit": "probability",
            }
        )
        writer.writerow({"metric": "steps", "value": len(states), "unit": "count"})


def _truth_ledger(scenario: Scenario, connectome: Connectome) -> str:
    sensory = "\n".join(
        (
            f"- {spec.modality}: {spec.mapper} ({spec.realism_level})"
            f" -> {', '.join(spec.target_regions)}"
        )
        for spec in scenario.stimuli
    )
    return f"""# Truth ledger: {scenario.scenario_id}

## Connectome

- Dataset: {connectome.dataset}
- Dynamics: {connectome.dynamics}
- Edges: {scenario.adaptation.connectome_edges}

## Scenario

- World backend: {scenario.world.backend}
- Task: {scenario.world.task}
- Observation Hz: {scenario.world.observation_hz}
- Adaptation preset: {scenario.adaptation.preset}
- Trainable params max: {scenario.adaptation.trainable_params_max}

## Sensory mappings

{sensory}

## Action decoder

- Decoder: {scenario.action.decoder}
- Target: {scenario.action.target}
- Action space: {", ".join(scenario.action.action_space)}

## Claim boundary

MaleCNS wiring is treated as the anatomical prior when real data is mounted.
This scaffold currently uses deterministic placeholder dynamics and generated state traces.
Runtime dynamics are synthetic unless a scenario names a validated model.
This run is not biological validation.
"""
