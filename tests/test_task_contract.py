import pytest

from cns_tinker.api import run
from cns_tinker.scenario.schema import load_scenario
from cns_tinker.task import validate_task


@pytest.mark.parametrize(
    "scope,frozen", [("readout", True), ("adapters", True), ("weights", False), ("topology", False)]
)
def test_training_scope_preserves_derivation_boundary(scope, frozen):
    task = validate_task(
        dict(
            schema_version=1,
            task="flight_sim",
            input="position",
            output="waypoint",
            objective="reach",
            training_scope=scope,
            seed=7,
            max_steps=2000,
            execution_status="running",
            graph_frozen=True,
        )
    )
    assert task["graph_frozen"] == frozen
    assert task["execution_status"] == "configuration_only"


def test_managed_mode_cannot_silently_execute_locally(tmp_path):
    scenario = load_scenario("recipes/00_no_train_loom_escape/scenario.yaml")
    with pytest.raises(NotImplementedError):
        run(scenario=scenario, mode="managed").execute(tmp_path / "no-run")
    assert not (tmp_path / "no-run").exists()
