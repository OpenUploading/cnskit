"""Portable task specifications; validation does not imply execution support."""

import json
from pathlib import Path

TASKS = {"flight_sim", "paper_market", "code_tools", "game_control", "custom"}
SCOPES = {
    "readout": (True, "readout_only"),
    "adapters": (True, "input_and_readout"),
    "weights": (False, "weights_and_dynamics"),
    "topology": (False, "derived_network"),
}


def validate_task(payload: dict) -> dict:
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported schema_version")
    if payload.get("task") not in TASKS:
        raise ValueError("unknown task")
    for field in ("input", "output", "objective"):
        value = payload.get(field)
        if not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError(f"{field} must be nonempty text of at most 2000 characters")
    scope = payload.get("training_scope")
    if scope not in SCOPES:
        raise ValueError("unknown training_scope")
    seed = payload.get("seed")
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("seed must be an integer from 0 to 2147483647")
    steps = payload.get("max_steps")
    if type(steps) is not int or not 1 <= steps <= 1000000:
        raise ValueError("max_steps must be an integer from 1 to 1000000")
    frozen, adaptation = SCOPES[scope]
    return {
        "schema_version": 1,
        "task": payload["task"],
        "input": payload["input"].strip(),
        "output": payload["output"].strip(),
        "objective": payload["objective"].strip(),
        "training_scope": scope,
        "seed": seed,
        "max_steps": steps,
        "graph_frozen": frozen,
        "adaptation": adaptation,
        "network_label": "fixed_graph" if frozen else "MaleCNS-derived (planned)",
        "execution_status": "configuration_only",
        "requirements": ["dataset_binding", "environment_adapter", "trainer", "evaluation"],
    }


def load_task(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task must be a JSON object")
    return validate_task(payload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate a task plan; does not launch a job")
    parser.add_argument("path")
    args = parser.parse_args()
    try:
        print(json.dumps(load_task(args.path), indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f"Invalid task: {error}\n")
