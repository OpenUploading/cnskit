from pathlib import Path

import numpy as np
import pytest

from cns_tinker.scenario.schema import load_scenario
from cns_tinker.session import Session


def session():
    return Session(load_scenario(Path(__file__).parents[1] / 'recipes/01_photon_saber_readout/scenario.yaml'))


def test_application_inputs_change_actions_and_reset_is_reproducible():
    run = session()
    start = run.state.activity.copy()
    for _ in range(60):
        left = run.step({'target_left': 1, 'target_right': 0, 'urgency': 1})
    for _ in range(60):
        right = run.step({'target_left': 0, 'target_right': 1, 'urgency': 1})
    assert left.action == 'swing_left'
    assert right.action == 'swing_right'
    np.testing.assert_array_equal(run.reset().activity, start)
    assert run.state.t_ms == 0


def test_bad_input_does_not_advance_and_snapshot_is_isolated():
    run = session()
    before = run.state.activity.copy()
    snapshot = run.state
    snapshot.activity[:] = 99
    np.testing.assert_array_equal(run.state.activity, before)
    with pytest.raises(ValueError):
        run.step({'target_left': float('nan')})
    assert run.state.t_ms == 0
