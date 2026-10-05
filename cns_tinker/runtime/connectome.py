from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from cns_tinker.sensory.adapters import Drive


@dataclass(frozen=True)
class RuntimeState:
    t_ms: float
    activity: np.ndarray


class Connectome:
    """Fixed connectome runtime facade.

    The current implementation is a deterministic placeholder dynamics engine. It is intentionally
    small and explicit so scenario, adapter, and evidence paths can be developed before private or
    licensed MaleCNS datasets are mounted.
    """

    def __init__(
        self,
        dataset: str = "male-cns-v1-placeholder",
        dynamics: str = "leaky-rate-v0",
        neuron_count: int = 512,
        seed: int = 7,
    ) -> None:
        if neuron_count < 64:
            raise ValueError("neuron_count must be at least 64 for region summaries")
        self.dataset = dataset
        self.dynamics = dynamics
        self.neuron_count = neuron_count
        self.seed = seed
        self.regions = (
            "retina",
            "lamina",
            "medulla",
            "lobula",
            "mushroom_body",
            "central_complex",
            "vnc",
            "task_channel",
        )

    def initial_state(self, seed: int | None = None) -> RuntimeState:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        activity = rng.normal(0.0, 0.015, self.neuron_count).astype(np.float32)
        return RuntimeState(t_ms=0.0, activity=activity)

    def step(self, state: RuntimeState, drive: Drive, dt_ms: float = 5.0) -> RuntimeState:
        if self.dynamics != "leaky-rate-v0":
            raise ValueError(f"unsupported dynamics: {self.dynamics}")
        dt = dt_ms / 1000.0
        activity = state.activity
        recurrent = (
            0.42 * np.roll(activity, 1)
            + 0.18 * np.roll(activity, 7)
            - 0.08 * np.roll(activity, -3)
        )
        current = self._drive_current(drive.channels)
        tau = 0.080
        next_activity = activity + (dt / tau) * (-activity + np.tanh(recurrent + current))
        return RuntimeState(t_ms=state.t_ms + dt_ms, activity=next_activity.astype(np.float32))

    def region_means(self, state: RuntimeState) -> dict[str, float]:
        chunks = np.array_split(state.activity, len(self.regions))
        return {
            region: float(chunk.mean())
            for region, chunk in zip(self.regions, chunks, strict=True)
        }

    def truth_metadata(self) -> dict[str, object]:
        return {
            "connectome_dataset": self.dataset,
            "dynamics": self.dynamics,
            "neuron_count": self.neuron_count,
            "seed": self.seed,
            "input_projection": "authored-task-feature-subpopulations-v1",
            "connectome_edges": "frozen_placeholder",
            "claim_boundary": (
                "real wiring plus synthetic dynamics when real MaleCNS data is mounted; "
                "not biological validation"
            ),
        }

    def _drive_current(self, channels: Mapping[str, float]) -> np.ndarray:
        current = np.zeros(self.neuron_count, dtype=np.float32)
        if not channels:
            return current
        chunks = np.array_split(np.arange(self.neuron_count), len(self.regions))
        region_to_idx = dict(zip(self.regions, chunks, strict=True))
        for name, value in channels.items():
            target = self._channel_region(name)
            idx = region_to_idx[target]
            # Authored feature layout, not anatomical. Preserve direction rather
            # than summing left + right into an identical scalar drive.
            task_features = ("synthetic_target_left", "synthetic_target_right", "synthetic_urgency")
            if name in task_features:
                idx = np.array_split(idx, 3)[task_features.index(name)]
            current[idx] += np.float32(value)
        return current

    def _channel_region(self, channel: str) -> str:
        if channel.startswith(("loom", "contrast", "uv", "green", "bearing")):
            return "lobula"
        if channel.startswith(("odor", "plume")):
            return "mushroom_body"
        if channel.startswith(("sugar", "water", "salt", "bitter", "taste")):
            return "vnc"
        if channel.startswith(("wind", "contact", "tilt")):
            return "central_complex"
        if channel.startswith(("song", "pulse")):
            return "mushroom_body"
        if channel.startswith(("target", "score", "synthetic")):
            return "task_channel"
        return "medulla"
