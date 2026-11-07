"""Explicit CPU thread budget; inference and training use the same CPU runtime."""
from __future__ import annotations
import os
import torch


def compute_summary():
    return {'device': 'cpu', 'logical_cpus': os.cpu_count() or 1,
            'torch_threads': torch.get_num_threads(),
            'torch_interop_threads': torch.get_num_interop_threads(),
            'torch_version': str(torch.__version__),
            'cuda_available': bool(torch.cuda.is_available()),
            'scope': 'CPU execution; utilization depends on batch size and workload'}


def configure_torch(threads=None):
    """Apply a process-wide budget, defaulting to at most four logical CPUs.

    Inter-op threads are intentionally left alone: PyTorch only allows changing
    that setting once, which is unsuitable for a runtime with multiple agents.
    """
    logical = os.cpu_count() or 1
    selected = min(4, logical) if threads is None else threads
    if type(selected) is not int or not 1 <= selected <= logical:
        raise ValueError(f'threads must be an integer in [1,{logical}]')
    if torch.get_num_threads() != selected:
        torch.set_num_threads(selected)
    return compute_summary()
