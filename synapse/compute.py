"""Device discovery and conservative runtime budgets for PyTorch workloads."""
from __future__ import annotations

import importlib.util
import os
from functools import lru_cache

import torch


def _optional_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _native_accelerator():
    """Return PyTorch's selected accelerator without importing vendor SDKs."""
    accelerator = getattr(torch, 'accelerator', None)
    if accelerator is None:
        return None
    try:
        device = accelerator.current_accelerator(check_available=True)
    except Exception:
        return None
    if device is None:
        return None
    try:
        device = torch.device(device)
    except (TypeError, RuntimeError, ValueError):
        return None
    return None if device.type == 'cpu' else device


@lru_cache(maxsize=1)
def accelerator_inventory():
    """Describe accelerators that the installed PyTorch runtime can see.

    Discovery is intentionally separate from use: the recurrent simulation
    still owns CPU state, while isolated training can request a probed device.
    """
    native = _native_accelerator()
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False
    mps = getattr(getattr(torch, 'backends', None), 'mps', None)
    try:
        mps_available = bool(mps is not None and mps.is_available())
    except Exception:
        mps_available = False
    xpu_backend = getattr(torch, 'xpu', None)
    try:
        xpu_available = bool(xpu_backend is not None and xpu_backend.is_available())
    except Exception:
        xpu_available = False

    xla_installed = _optional_module('torch_xla')
    torch_tpu_installed = _optional_module('torch_tpu')
    reported_type = native.type if native is not None else None
    xla_hardware = None
    if xla_installed:
        try:
            import torch_xla.runtime as xla_runtime
            xla_hardware = str(xla_runtime.device_type()).upper()
        except Exception:
            xla_hardware = None
    tpu_available = reported_type == 'tpu' or (
        torch_tpu_installed and reported_type == 'xla'
    ) or xla_hardware == 'TPU'
    xla_accelerator_available = xla_hardware in {'TPU', 'GPU'} or reported_type == 'xla'

    candidates = []
    if native is not None:
        candidates.append(str(native))
    for available, name in ((cuda_available, 'cuda'), (mps_available, 'mps'), (xpu_available, 'xpu')):
        if available and not any(item.split(':', 1)[0] == name for item in candidates):
            candidates.append(name)
    if tpu_available and not any(item.split(':', 1)[0] == 'tpu' for item in candidates):
        candidates.append('tpu' if reported_type != 'xla' else 'xla')
    if xla_accelerator_available and not any(item.split(':', 1)[0] == 'xla' for item in candidates):
        candidates.append('xla')

    return {
        'cuda_available': cuda_available,
        'cuda_build': getattr(torch.version, 'cuda', None),
        'mps_available': mps_available,
        'xpu_available': xpu_available,
        'tpu_available': tpu_available,
        'torch_tpu_installed': torch_tpu_installed,
        'torch_xla_installed': xla_installed,
        'xla_hardware': xla_hardware,
        'training_device_candidates': candidates,
    }


def _probe_training_device(device_name: str) -> bool:
    """Check a tiny forward/backward/optimizer cycle before moving real state."""
    device = torch.device(device_name)
    if device.type == 'cpu':
        return True
    model = torch.nn.Linear(4, 2, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sample = torch.ones((2, 4), dtype=torch.float32, device=device)
    loss = model(sample).square().mean()
    if not bool(torch.isfinite(loss).item()):
        return False
    loss.backward()
    optimizer.step()
    # Synchronize the tiny probe so asynchronous failures surface before fitting.
    return bool(torch.isfinite(loss.detach().to('cpu')).item())


def resolve_training_device(preferred: str = 'auto'):
    """Select an operational PyTorch device, otherwise retain the CPU path.

    TPU/XLA and vendor accelerators are optional; discovery or a failed probe
    never prevents the existing CPU implementation from running.
    """
    if not isinstance(preferred, str) or preferred not in {'auto', 'cpu'}:
        raise ValueError("preferred must be 'auto' or 'cpu'")
    inventory = accelerator_inventory()
    if preferred == 'cpu':
        return 'cpu', {'requested': preferred, 'selected': 'cpu', 'fallback_reason': None}

    errors = []
    for candidate in inventory['training_device_candidates']:
        try:
            if _probe_training_device(candidate):
                return candidate, {'requested': preferred, 'selected': candidate, 'fallback_reason': None}
        except Exception as exc:
            errors.append(f'{candidate}: {type(exc).__name__}: {exc}')
    if errors:
        reason = '; '.join(errors)
    elif inventory['training_device_candidates']:
        reason = 'detected accelerators did not pass the training probe'
    else:
        reason = 'no usable PyTorch accelerator detected'
    return 'cpu', {'requested': preferred, 'selected': 'cpu', 'fallback_reason': reason}


def available_memory_bytes():
    """Return OS-reported free physical memory when available."""
    if os.name == 'nt':
        try:
            import ctypes

            class _MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
        except (AttributeError, OSError, TypeError, ValueError):
            return None
    if hasattr(os, 'sysconf'):
        try:
            return int(os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE'))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return None


def safe_worker_count(requested: int, neurons: int = 9000) -> int:
    """Bound cloned-world workers by CPUs and free RAM to avoid burst OOMs."""
    cpus = os.cpu_count() or 1
    if type(requested) is not int or not 1 <= requested <= 4:
        raise ValueError('workers must be an integer in [1,4]')
    if type(neurons) is not int or neurons < 1:
        raise ValueError('neurons must be a positive integer')
    memory = available_memory_bytes()
    if memory is None:
        return min(requested, 4, cpus)
    mib = 1024 * 1024
    reserve = 1536 * mib
    per_worker = max(384 * mib, min(1536 * mib, neurons * 65536))
    memory_workers = max(1, (memory - reserve) // per_worker)
    return max(1, min(requested, 4, cpus, int(memory_workers)))


def compute_summary():
    inventory = accelerator_inventory()
    logical = os.cpu_count() or 1
    return {
        # The integrated world/graph/telemetry path currently keeps mutable
        # simulator state on CPU; eligible offline fitting may use an accelerator.
        'device': 'cpu',
        'logical_cpus': logical,
        'torch_threads': torch.get_num_threads(),
        'torch_interop_threads': torch.get_num_interop_threads(),
        'torch_version': str(torch.__version__),
        **inventory,
        'scope': 'CPU recurrent simulation; isolated neural-graft fitting uses a probed PyTorch accelerator when available and retries on CPU if needed',
    }


def configure_torch(threads=None):
    """Set a validated intra-op CPU budget without risking a server crash.

    Max Mode passes the visible CPU count (up to 32); the regular runtime keeps
    its smaller default. Inter-op threads stay untouched because PyTorch only
    allows changing them once per process.
    """
    logical = os.cpu_count() or 1
    selected = min(4, logical) if threads is None else threads
    if type(selected) is not int or not 1 <= selected <= logical:
        raise ValueError(f'threads must be an integer in [1,{logical}]')
    error = None
    if torch.get_num_threads() != selected:
        try:
            torch.set_num_threads(selected)
        except RuntimeError as exc:
            # Keep an already-running shared server alive; expose the actual
            # setting so the Max Mode receipt cannot claim a budget it missed.
            error = f'{type(exc).__name__}: {exc}'
    result = compute_summary()
    result['requested_torch_threads'] = selected
    result['thread_configuration_error'] = error
    return result
