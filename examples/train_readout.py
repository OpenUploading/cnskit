"""Fit an authored target on synthetic episodes; this is not a MaleCNS benchmark."""
import argparse
from pathlib import Path
import numpy as np
from cns_tinker.readout import RidgeReadout
from cns_tinker.scenario.schema import load_scenario
from cns_tinker.session import Session


def episode(seed):
    scenario = load_scenario(Path(__file__).resolve().parents[1]/"recipes/01_photon_saber_readout/scenario.yaml")
    session = Session(scenario, seed=7)
    rng = np.random.default_rng(seed)
    x, y = [], []
    for step in range(120):
        if step % 30 == 0: left = float(rng.uniform(.1,.9))
        session.step({"target_left":left,"target_right":1-left,"urgency":1.0})
        x.append(session.state.activity[::16])
        y.append([2*left-1])
    return np.asarray(x),np.asarray(y)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",default="readout.json")
    args=parser.parse_args()
    training=[episode(seed) for seed in (10,11,12,13)]
    x=np.concatenate([p[0] for p in training]); y=np.concatenate([p[1] for p in training])
    test_x,test_y=episode(20)
    names=[f"synthetic_{i}" for i in range(0,512,16)]
    model=RidgeReadout.fit(x,y,feature_names=names,output_names=["steering_target"],alpha=1.)
    model.save(args.out)
    restored=RidgeReadout.load(args.out)
    print("Held-out episode MSE:",restored.evaluate(test_x,test_y,feature_names=names))
    print("Training-mean baseline MSE:",float(np.mean((test_y-y.mean(axis=0))**2)))
    print("Checkpoint:",args.out,"; synthetic demonstration, no connectome fine-tuning")


if __name__ == "__main__": main()
