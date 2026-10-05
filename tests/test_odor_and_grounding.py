from pathlib import Path

from cns_tinker.evals import score_bundle
from cns_tinker.runner import run_scenario
from cns_tinker.scenario.schema import load_scenario


def test_odor_liveproof_recipe_runs_and_scores(tmp_path: Path) -> None:
    scenario = load_scenario("recipes/05_odor_liveproof/scenario.yaml")
    assert {spec.modality for spec in scenario.stimuli} == {"olfactory"}

    run_scenario(scenario, tmp_path)
    score = score_bundle(tmp_path)

    assert score.label == "inspectability_grounding"
    assert score.value == 1.0
    assert score.notes == ()
