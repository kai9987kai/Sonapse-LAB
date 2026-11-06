import torch
import numpy as np
from synapse.astrocyte import AstrocyteNetwork

def test_astrocyte_initialization():
    astro = AstrocyteNetwork(num_domains=3)
    assert astro.num_domains == 3
    assert len(astro.domains) == 3
    summary = astro.summary()
    assert summary['num_domains'] == 3
    assert 0.0 < summary['mean_calcium'] < 1.0

def test_astrocyte_calcium_wave_and_gain():
    astro = AstrocyteNetwork(num_domains=3)
    # Simulate high neural rates
    high_rates = torch.full((300,), 2.5)
    gains = astro.step(high_rates)
    assert len(gains) == 3
    # High activity should stimulate calcium influx
    summary = astro.summary()
    assert summary['mean_calcium'] > 0.1
    # BCM threshold should slide upward
    assert summary['mean_bcm_threshold'] > 0.5

def test_astrocyte_state_roundtrip():
    astro = AstrocyteNetwork(num_domains=3)
    astro.step(torch.rand(300))
    state = astro.export_state()
    astro2 = AstrocyteNetwork(num_domains=3)
    astro2.import_state(state)
    assert np.allclose(astro.calcium_levels.numpy(), astro2.calcium_levels.numpy())
    assert np.allclose(astro.bcm_thresholds.numpy(), astro2.bcm_thresholds.numpy())
