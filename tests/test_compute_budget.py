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


def test_compute_inventory_keeps_core_runtime_cpu_and_exposes_optional_devices():
    from synapse.compute import accelerator_inventory, compute_summary, resolve_training_device
    summary = compute_summary()
    inventory = accelerator_inventory()
    assert summary['device'] == 'cpu'
    assert summary['training_device_candidates'] == inventory['training_device_candidates']
    assert isinstance(inventory['cuda_available'], bool)
    assert isinstance(inventory['mps_available'], bool)
    assert isinstance(inventory['tpu_available'], bool)
    device, selection = resolve_training_device('cpu')
    assert device == 'cpu'
    assert selection['selected'] == 'cpu'


def test_worker_budget_respects_free_memory(monkeypatch):
    from synapse import compute
    monkeypatch.setattr(compute, 'available_memory_bytes', lambda: 1024 * 1024 * 1024)
    assert compute.safe_worker_count(4, neurons=9000) == 1
