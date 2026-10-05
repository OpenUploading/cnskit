from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Drive:
    channels: Mapping[str, float]
    realism_level: str
    target_regions: tuple[str, ...]
    summary: str


@dataclass(frozen=True)
class SensorySpec:
    modality: str
    mapper: str
    realism_level: str = "schematic"
    target_regions: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, modality: str, payload: Mapping[str, object]) -> SensorySpec:
        regions = payload.get("target_regions", ())
        if isinstance(regions, str):
            target_regions: tuple[str, ...] = (regions,)
        else:
            target_regions = tuple(str(item) for item in regions)  # type: ignore[arg-type]
        return cls(
            modality=modality,
            mapper=str(payload["mapper"]),
            realism_level=str(payload.get("realism_level", "schematic")),
            target_regions=target_regions,
        )


class SensoryAdapter:
    def __init__(self, spec: SensorySpec) -> None:
        self.spec = spec

    def encode(self, observation: Mapping[str, float]) -> Drive:
        raise NotImplementedError


class VisualLoomFlow(SensoryAdapter):
    def encode(self, observation: Mapping[str, float]) -> Drive:
        expansion = float(observation.get("expansion_rate", observation.get("loom", 0.0)))
        contrast = float(observation.get("contrast", 0.0))
        bearing = float(observation.get("bearing", 0.0))
        return Drive(
            channels={
                "loom_expansion": expansion,
                "contrast": contrast,
                "bearing_abs": abs(bearing) * 0.15,
            },
            realism_level=self.spec.realism_level,
            target_regions=self.spec.target_regions,
            summary="optic-flow and looming drive",
        )


class GustatoryContact(SensoryAdapter):
    def encode(self, observation: Mapping[str, float]) -> Drive:
        return Drive(
            channels={
                "sugar_contact": float(observation.get("sugar", 0.0)),
                "water_contact": float(observation.get("water", 0.0)),
                "salt_contact": float(observation.get("salt", 0.0)),
                "bitter_contact": float(observation.get("bitter", 0.0)),
            },
            realism_level=self.spec.realism_level,
            target_regions=self.spec.target_regions,
            summary="gustatory contact drive",
        )


class OlfactoryPlume(SensoryAdapter):
    def encode(self, observation: Mapping[str, float]) -> Drive:
        concentration = float(observation.get("odor_concentration", 0.0))
        gradient = float(observation.get("odor_gradient", 0.0))
        hit = float(observation.get("odor_hit", 0.0))
        return Drive(
            channels={
                "odor_concentration": concentration,
                "odor_gradient": gradient,
                "plume_hit": hit,
            },
            realism_level=self.spec.realism_level,
            target_regions=self.spec.target_regions,
            summary="olfactory plume drive",
        )


class SyntheticArcadeTarget(SensoryAdapter):
    def encode(self, observation: Mapping[str, float]) -> Drive:
        target_left = float(observation.get("target_left", 0.0))
        target_right = float(observation.get("target_right", 0.0))
        urgency = float(observation.get("urgency", 0.0))
        return Drive(
            channels={
                "synthetic_target_left": target_left,
                "synthetic_target_right": target_right,
                "synthetic_urgency": urgency,
            },
            realism_level=self.spec.realism_level,
            target_regions=self.spec.target_regions,
            summary="synthetic task channel for arcade demos",
        )


class MechanosensoryWind(SensoryAdapter):
    def encode(self, observation: Mapping[str, float]) -> Drive:
        return Drive(
            channels={
                "wind_speed": float(observation.get("wind_speed", 0.0)),
                "contact_load": float(observation.get("contact", 0.0)),
                "tilt": float(observation.get("tilt", 0.0)),
            },
            realism_level=self.spec.realism_level,
            target_regions=self.spec.target_regions,
            summary="wind and contact mechanosensory drive",
        )


def build_sensory_adapter(spec: SensorySpec) -> SensoryAdapter:
    key = (spec.modality, spec.mapper)
    if key == ("visual", "optic_flow_and_loom"):
        return VisualLoomFlow(spec)
    if key == ("gustatory", "contact_chemistry"):
        return GustatoryContact(spec)
    if key == ("olfactory", "plume_receptor_projection"):
        return OlfactoryPlume(spec)
    if key == ("synthetic", "arcade_target"):
        return SyntheticArcadeTarget(spec)
    if key == ("mechanosensory", "wind_contact"):
        return MechanosensoryWind(spec)
    raise ValueError(f"unsupported sensory adapter: {spec.modality}/{spec.mapper}")


def merge_drives(drives: Sequence[Drive]) -> Drive:
    channels: dict[str, float] = {}
    regions: list[str] = []
    summaries: list[str] = []
    levels: list[str] = []
    for drive in drives:
        for name, value in drive.channels.items():
            channels[name] = channels.get(name, 0.0) + value
        regions.extend(drive.target_regions)
        summaries.append(drive.summary)
        levels.append(drive.realism_level)
    return Drive(
        channels=channels,
        realism_level="+".join(sorted(set(levels))) if levels else "none",
        target_regions=tuple(dict.fromkeys(regions)),
        summary="; ".join(summaries),
    )
