from pathlib import Path

from cns_tinker.api import run
from cns_tinker.scenario.schema import load_scenario


def test_run_is_the_primary_local_sdk_entrypoint(tmp_path: Path) -> None:
    scenario = load_scenario(Path("recipes/00_no_train_loom_escape/scenario.yaml"))

    result = run(scenario=scenario).execute(tmp_path / "bundle")

    assert result.output_dir == tmp_path / "bundle"


def test_run_rejects_unknown_execution_mode() -> None:
    scenario = load_scenario(Path("recipes/00_no_train_loom_escape/scenario.yaml"))

    try:
        run(scenario=scenario, mode="everywhere")
    except ValueError as error:
        assert "mode" in str(error)
    else:
        raise AssertionError("expected invalid execution mode to fail")
