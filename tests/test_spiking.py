"""Tests for LIF and AdEx Spiking Dynamics and Neuromorphic SynOps."""
import pytest
import torch
from synapse.spiking import LIFSpikingLayer, AdExSpikingLayer, benchmark_spiking_vs_rate


def test_lif_spiking_layer():
    layer = LIFSpikingLayer(in_features=8, out_features=16)
    layer.reset_state()
    x = torch.randn(8)
    s, v = layer(x)
    assert s.shape == (16,)
    assert v.shape == (16,)
    assert set(s.unique().tolist()).issubset({0.0, 1.0})
    assert layer.total_synops >= 0


def test_adex_spiking_layer():
    layer = AdExSpikingLayer(in_features=8, out_features=16)
    layer.reset_state()
    x = torch.randn(8)
    s, v = layer(x)
    assert s.shape == (16,)
    assert v.shape == (16,)
    assert set(s.unique().tolist()).issubset({0.0, 1.0})


def test_benchmark_spiking_vs_rate():
    res = benchmark_spiking_vs_rate(num_neurons=32, in_features=8, steps=30, seed=42)
    assert 'rate' in res
    assert 'lif_spiking' in res
    assert 'adex_spiking' in res
    assert res['rate']['total_macs'] > 0
    assert res['lif_spiking']['sparsity'] > 0.5
    assert res['lif_spiking']['synops_reduction_ratio'] > 1.0
