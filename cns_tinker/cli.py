from __future__ import annotations

import argparse
import json
from pathlib import Path

from cns_tinker.runner import run_scenario
from cns_tinker.scenario.schema import load_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cns-tinker")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a scenario YAML/JSON file")
    validate.add_argument("scenario")

    run = sub.add_parser("run", help="run a scenario and export an evidence bundle")
    run.add_argument("scenario")
    run.add_argument("--out", default="runs/latest")

    real = sub.add_parser("real-run", help="record explicit body-ID currents on a local MaleCNS graph")
    real.add_argument("--graph", required=True)
    real.add_argument("--inputs", required=True, help="JSON array of per-step body-ID/current objects")
    real.add_argument("--readout-type", required=True, help="exact official cell type, e.g. DNp01")
    real.add_argument("--workers", type=int, default=1)
    real.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.command == "real-run":
        from cns_tinker.runtime.malecns import MaleCNSSession

        path = Path(args.inputs)
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Input JSON exceeds 32 MiB")
        frames = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(frames, list) or not 1 <= len(frames) <= 10000:
            raise ValueError("Inputs must contain 1 to 10000 frames")
        sequence = []
        for frame in frames:
            if not isinstance(frame, dict):
                raise ValueError("Each input frame must be an object")
            currents = {}
            for key, value in frame.items():
                if str(int(key)) != key or type(value) not in (int, float):
                    raise ValueError("Use integer body-ID keys and numeric current values")
                currents[int(key)] = value
            sequence.append(currents)
        with MaleCNSSession(args.graph, workers=args.workers) as session:
            ids = session.select(cell_type=args.readout_type)
            if not ids:
                raise ValueError("No traced cells match the requested readout type")
            out = session.record(sequence, readout_ids=ids, out=args.out)
        print(f"wrote real-graph run: {out}; synthetic dynamics, not biological validation")
        return 0
    if args.command == "validate":
        scenario = load_scenario(args.scenario)
        print(f"valid scenario: {scenario.scenario_id}")
        return 0
    if args.command == "run":
        result = run_scenario(args.scenario, Path(args.out))
        print(f"wrote evidence bundle: {result.output_dir}")
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
