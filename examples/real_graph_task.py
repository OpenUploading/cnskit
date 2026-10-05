"""Explicit current injection; requires a prepared real MaleCNS graph."""
import argparse
from cns_tinker.runtime.malecns import MaleCNSSession


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    args = parser.parse_args()
    with MaleCNSSession(args.graph) as session:
        inputs = session.select(cell_type="LC4")
        outputs = session.select(cell_type="DNp01")
        if not inputs or not outputs:
            raise ValueError("Prepared graph lacks LC4 inputs or DNp01 readouts")
        for i in range(120):
            session.step(dict.fromkeys(inputs, 1.0) if 20 <= i < 80 else {})
            if i in (19, 79, 119):
                print(f"{session.state.t_ms:.0f} ms", session.read(outputs))
    print("Real wiring + synthetic dynamics; readout activity is not validated behavior.")


if __name__ == "__main__":
    main()
