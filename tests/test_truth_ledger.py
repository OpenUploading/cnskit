from pathlib import Path

from cns_tinker.runner import run_scenario


def test_truth_ledger_contains_claim_boundary(tmp_path: Path) -> None:
    run_scenario("recipes/01_photon_saber_readout/scenario.yaml", tmp_path)
    ledger = (tmp_path / "truth_ledger.md").read_text(encoding="utf-8")

    assert "not biological validation" in ledger
    assert "synthetic" in ledger
    assert "photon_saber_readout_v0" in ledger
