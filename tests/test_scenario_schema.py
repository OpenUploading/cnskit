from pathlib import Path

from cns_tinker.scenario.schema import load_scenario


def test_load_loom_recipe() -> None:
    scenario = load_scenario(Path("recipes/00_no_train_loom_escape/scenario.yaml"))

    assert scenario.scenario_id == "loom_escape_v0"
    assert scenario.world.backend == "unreal"
    assert scenario.adaptation.connectome_edges == "frozen"
    assert {spec.modality for spec in scenario.stimuli} == {"visual", "mechanosensory"}
