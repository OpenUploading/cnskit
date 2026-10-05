from pathlib import Path

from cns_tinker.runner import run_scenario


def test_run_exports_evidence_bundle(tmp_path: Path) -> None:
    result = run_scenario("recipes/00_no_train_loom_escape/scenario.yaml", tmp_path)

    assert result.evidence.manifest_path.exists()
    assert (tmp_path / "scenario.yaml").exists()
    assert (tmp_path / "truth_ledger.md").exists()
    assert (tmp_path / "stimulus_trace.csv").exists()
    assert (tmp_path / "action_trace.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "neural_trace.npz").exists()
