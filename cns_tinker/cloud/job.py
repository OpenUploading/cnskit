from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CloudJobSpec:
    scenario_id: str
    provider: str
    gpu_class: str
    region: str
    max_minutes: int

    def queue_name(self) -> str:
        return f"{self.provider}:{self.region}:{self.gpu_class}"
