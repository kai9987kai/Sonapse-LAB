"""The CPU budget is explicit, validated, and reports the active torch setting."""
import os
import pytest
import torch


def test_configuring_cpu_budget_applies_and_reports_actual_threads():
    from synapse.compute import configure_torch
    before = torch.get_num_threads()
    try:
        result = configure_torch(threads=1)
        assert result['torch_threads'] == torch.get_num_threads() == 1
        assert result['logical_cpus'] == (os.cpu_count() or 1)
        assert result['device'] == 'cpu'
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize('value', [True, 0, -1, 1.5, 100000])
def test_invalid_cpu_budget_cannot_mutate_process_setting(value):
    from synapse.compute import configure_torch
    before = torch.get_num_threads()
    with pytest.raises(ValueError):
        configure_torch(value)
    assert torch.get_num_threads() == before
