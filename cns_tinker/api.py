from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cns_tinker.actions.adapters import ActionSpec
from cns_tinker.runner import RunResult, run_scenario
from cns_tinker.scenario.schema import Scenario
from cns_tinker.sensory.adapters import SensorySpec


@dataclass(frozen=True)
class World:
    backend: str
    task: str
    observation_hz: int = 60
    prompt: str | None = None

    @classmethod
    def from_prompt(cls, prompt: str, preset: str = "arena") -> World:
        lowered = prompt.lower()
        if "saber" in lowered or "rhythm" in lowered:
            return cls(backend="unreal", task="rhythm_saber", prompt=prompt)
        if "market" in lowered or "stock" in lowered:
            return cls(backend="unreal", task="market_arena", prompt=prompt)
        if "loom" in lowered or "escape" in lowered:
            return cls(backend="unreal", task="loom_escape", prompt=prompt)
        return cls(backend="unreal", task=preset, prompt=prompt)


class SensoryMap:
    @staticmethod
    def visual_loom_flow(realism_level: str = "grounded") -> SensorySpec:
        return SensorySpec(
            modality="visual",
            mapper="optic_flow_and_loom",
            realism_level=realism_level,
            target_regions=("retina", "lamina", "medulla", "lobula"),
        )

    @staticmethod
    def synthetic_arcade_target(realism_level: str = "schematic") -> SensorySpec:
        return SensorySpec(
            modality="synthetic",
            mapper="arcade_target",
            realism_level=realism_level,
            target_regions=("task_channel",),
        )


@dataclass(frozen=True)
class Agent:
    scenario: Scenario

    def play(
        self,
        render: str = "headless",
        brain_view: bool = True,
        out: str | Path = "runs/sdk",
    ) -> RunResult:
        _ = render, brain_view
        return run_scenario(self.scenario, out)


@dataclass(frozen=True)
class Run:
    """An immutable SDK handle for one local or managed runtime invocation.

    The current scaffold executes locally. A cloud-backed implementation can retain the
    same handle shape while replacing ``run_id`` with a remote immutable run identifier.
    """

    scenario: Scenario
    mode: str = "local"

    def execute(self, out: str | Path = "runs/sdk") -> RunResult:
        if self.mode != "local":
            raise NotImplementedError("Managed execution is not connected; use mode='local'.")
        return run_scenario(self.scenario, out)


def run(*, scenario: Scenario, mode: str = "local") -> Run:
    """Prepare one MaleCNS runtime invocation.

    ``run()`` is the primary integration surface: the caller owns its world and
    adapters, while CNS Tinker owns the versioned runtime contract and evidence bundle.
    ``mode='managed'`` is reserved for the future hosted scheduler.
    """

    if mode not in {"local", "managed"}:
        raise ValueError("mode must be 'local' or 'managed'")
    return Run(scenario=scenario, mode=mode)


@dataclass(frozen=True)
class LocalTuneJob:
    scenario: Scenario
    metadata: dict[str, Any]

    def deploy(self) -> Agent:
        return Agent(self.scenario)


def tune(
    *,
    brain: Any,
    world: World,
    sensory: SensorySpec,
    action: ActionSpec,
    adaptation: str,
    budget: str,
    evidence: bool = True,
) -> LocalTuneJob:
    """Create an explicit adapter/downstream-head tuning job.

    This remains secondary to :func:`run`: tuning does not alter the fixed
    connectome graph in this product model.
    """
    scenario = Scenario.from_parts(
        scenario_id=f"{world.task}_sdk",
        world=world,
        sensory=(sensory,),
        action=action,
        adaptation_preset=adaptation,
        evidence_enabled=evidence,
    )
    return LocalTuneJob(
        scenario=scenario,
        metadata={
            "brain": getattr(brain, "dataset", "unknown"),
            "dynamics": getattr(brain, "dynamics", "unknown"),
            "budget": budget,
            "mode": "local_stub",
        },
    )
