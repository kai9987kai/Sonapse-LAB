import json
import numpy as np


def test_nonfinite_model_forecast_abstains_with_serializable_diagnostics():
    from synapse.experience import ExperienceWorldModel
    from synapse.experience_planner import ExperienceMPC
    model = ExperienceWorldModel(state_dim=2, action_dim=1)
    state = model.export_state()
    state['weights'][:, 1, -1] = 1e308
    state['samples'] = 40
    state['member_updates'][:] = 40
    state['reward_error_ema'] = state['delta_error_ema'] = .01
    state['baseline_error_ema'] = state['delta_baseline_error_ema'] = 1.
    model.import_state(state)
    with np.errstate(over='ignore', invalid='ignore'):
        action, diagnostics = ExperienceMPC(model, horizon=3).plan([10, 0], [0])
    np.testing.assert_array_equal(action, [0])
    assert not diagnostics['used']
    json.dumps(diagnostics, allow_nan=False)
