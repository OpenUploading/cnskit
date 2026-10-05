from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GroundingScore:
    value: float
    max_value: float
    label: str
    notes: tuple[str, ...]


def score_bundle(bundle_dir: str | Path) -> GroundingScore:
    """Score whether a run bundle is inspectable, not whether it is biologically validated."""

    root = Path(bundle_dir)
    required = (
        "manifest.json",
        "scenario.yaml",
        "truth_ledger.md",
        "stimulus_trace.csv",
        "action_trace.csv",
        "metrics.csv",
        "neural_trace.npz",
    )
    notes: list[str] = []
    points = 0.0
    for name in required:
        if (root / name).exists():
            points += 1.0
        else:
            notes.append(f"missing {name}")

    metrics_path = root / "metrics.csv"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            metrics = {row["metric"]: row["value"] for row in csv.DictReader(handle)}
        if "mean_action_confidence" in metrics:
            points += 1.0
        else:
            notes.append("missing mean_action_confidence")

    ledger_path = root / "truth_ledger.md"
    ledger = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else ""
    if "not biological validation" in ledger:
        points += 1.0
    else:
        notes.append("truth ledger lacks biological-validation boundary")

    max_value = len(required) + 2.0
    return GroundingScore(
        value=round(points / max_value, 4),
        max_value=1.0,
        label="inspectability_grounding",
        notes=tuple(notes),
    )
