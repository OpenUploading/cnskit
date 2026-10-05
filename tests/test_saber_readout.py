import numpy as np
from cns_tinker.runtime.connectome import Connectome
from cns_tinker.sensory.adapters import Drive
from cns_tinker.actions.adapters import ActionSpec, build_action_adapter


def test_direction_survives_projection_and_state_readout():
    runtime = Connectome()
    decoder = build_action_adapter(ActionSpec('synthetic_saber_readout_v1', ('swing_left', 'swing_right')))
    def drive(left):
        return Drive({'synthetic_target_left': float(left), 'synthetic_target_right': float(not left)}, 'schematic', (), 'test')
    initial = runtime.initial_state()
    a = runtime.step(initial, drive(True))
    b = runtime.step(initial, drive(False))
    assert not np.allclose(a.activity, b.activity)
    state = initial
    for left in (True, False, True):
        for _ in range(60):
            state = runtime.step(state, drive(left), dt_ms=1000/60)
        assert decoder.decode(state).action == ('swing_left' if left else 'swing_right')
    # Removing input must actually decay the runtime; decoder is not reading a target flag.
    before = np.linalg.norm(state.activity)
    for _ in range(120):
        state = runtime.step(state, Drive({}, 'none', (), ''), dt_ms=1000/60)
    assert np.linalg.norm(state.activity) < before * .01
