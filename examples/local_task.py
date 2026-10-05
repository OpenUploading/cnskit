"""Run from the checkout root after pip install -e .; no dataset required."""
from pathlib import Path
from cns_tinker.scenario.schema import load_scenario
from cns_tinker.session import Session


def main():
    root = Path(__file__).resolve().parents[1]
    scenario = load_scenario(root / "recipes/01_photon_saber_readout/scenario.yaml")
    session = Session(scenario, seed=7)
    previous = None
    for i in range(240):
        left = (i // 60) % 2 == 0
        output = session.step({"target_left": float(left), "target_right": float(not left), "urgency": 1.0})
        if output.action != previous:
            print(f"{session.state.t_ms:8.1f} ms  {output.action}")
            previous = output.action
    print("Synthetic preview complete; no trained policy or anatomical graph was used.")


if __name__ == "__main__":
    main()
