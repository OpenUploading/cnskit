from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from cns_tinker.runtime.connectome import RuntimeState


@dataclass(frozen=True)
class ActionOutput:
    action: str
    confidence: float
    values: Mapping[str, float]


@dataclass(frozen=True)
class ActionSpec:
    decoder: str
    action_space: tuple[str, ...]
    target: str = "gamepad"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ActionSpec:
        space = payload.get("action_space", ())
        return cls(
            decoder=str(payload["decoder"]),
            action_space=tuple(str(item) for item in space),  # type: ignore[arg-type]
            target=str(payload.get("target", "gamepad")),
        )


class ActionAdapter:
    def __init__(self, spec: ActionSpec) -> None:
        self.spec = spec

    def decode(self, state: RuntimeState) -> ActionOutput:
        raise NotImplementedError


class DiscreteGamepad(ActionAdapter):
    def decode(self, state: RuntimeState) -> ActionOutput:
        chunks = np.array_split(state.activity, max(len(self.spec.action_space), 1))
        logits = np.array([float(chunk.mean()) for chunk in chunks], dtype=np.float32)
        if not self.spec.action_space:
            return ActionOutput(action="noop", confidence=1.0, values={})
        idx = int(np.argmax(logits))
        shifted = logits - float(logits.max())
        probs = np.exp(shifted)
        probs = probs / float(probs.sum())
        values = {name: float(probs[i]) for i, name in enumerate(self.spec.action_space)}
        return ActionOutput(
            action=self.spec.action_space[idx],
            confidence=float(probs[idx]),
            values=values,
        )


class SyntheticSaberReadout(ActionAdapter):
    """Authored readout from evolved task populations; no observation shortcut."""

    def decode(self, state: RuntimeState) -> ActionOutput:
        task = np.array_split(state.activity, 8)[-1]
        left, right, _ = [float(x.mean()) for x in np.array_split(task, 3)]
        logits = np.array([left, right]) * 4.0
        weights = np.exp(logits - logits.max())
        weights /= weights.sum()
        index = int(np.argmax(weights))
        actions = ("swing_left", "swing_right")
        return ActionOutput(actions[index], float(weights[index]),
                            dict(zip(actions, map(float, weights), strict=True)))


class UnrealFlyEthology(ActionAdapter):
    def decode(self, state: RuntimeState) -> ActionOutput:
        thirds = np.array_split(state.activity, 3)
        drive = [float(chunk.mean()) for chunk in thirds]
        actions = self.spec.action_space or ("idle", "approach", "escape")
        idx = int(np.argmax(np.array(drive[: len(actions)], dtype=np.float32)))
        confidence = min(1.0, 0.5 + abs(drive[idx]) * 5.0)
        return ActionOutput(
            action=actions[idx],
            confidence=confidence,
            values=dict(zip(actions, drive, strict=False)),
        )


def build_action_adapter(spec: ActionSpec) -> ActionAdapter:
    if spec.decoder == "synthetic_saber_readout_v1":
        if spec.action_space != ("swing_left", "swing_right"):
            raise ValueError("synthetic saber readout requires swing_left, swing_right")
        return SyntheticSaberReadout(spec)
    if spec.decoder == "vnc_to_gamepad":
        return DiscreteGamepad(spec)
    if spec.decoder == "ethology_event":
        return UnrealFlyEthology(spec)
    raise ValueError(f"unsupported action decoder: {spec.decoder}")


class ActionMap:
    @staticmethod
    def discrete_gamepad(actions: Sequence[str]) -> ActionSpec:
        return ActionSpec(decoder="vnc_to_gamepad", action_space=tuple(actions), target="gamepad")

    @staticmethod
    def unreal_fly_ethology(actions: Sequence[str] = ("idle", "approach", "escape")) -> ActionSpec:
        return ActionSpec(
            decoder="ethology_event",
            action_space=tuple(actions),
            target="unreal_fly",
        )
