"""Run a deterministic scale-shift audit of the online residual calibrator."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synapse.experience import AdaptiveResidualCalibrator


def run_audit(seed=20261111, phase_length=800):
    """Compare fixed and adaptive intervals on one synthetic residual stream."""
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if type(phase_length) is not int or phase_length < 100:
        raise ValueError('phase_length must be an integer of at least 100')

    rng = np.random.default_rng(seed)
    scores = np.concatenate((np.abs(rng.normal(0.0, 1.0, phase_length)),
                             np.abs(rng.normal(0.0, 2.0, phase_length))))
    calibrator = AdaptiveResidualCalibrator()
    fixed_threshold = 1.6448536269514722
    observations = [[], []]
    for index, score in enumerate(scores):
        phase = 0 if index < phase_length else 1
        adaptive = calibrator.observe(float(score))
        if adaptive['active']:
            observations[phase].append({
                'fixed_miss': bool(score > fixed_threshold),
                'adaptive_miss': bool(adaptive['miss']),
                'adaptive_threshold': float(adaptive['threshold']),
            })

    phases = []
    for name, values in zip(('stationary', 'scale_shift'), observations):
        phases.append({
            'phase': name,
            'evaluated_predictions': len(values),
            'fixed_threshold': fixed_threshold,
            'fixed_misses': sum(value['fixed_miss'] for value in values),
            'fixed_miscoverage': float(np.mean([value['fixed_miss'] for value in values])),
            'adaptive_misses': sum(value['adaptive_miss'] for value in values),
            'adaptive_miscoverage': float(np.mean([value['adaptive_miss'] for value in values])),
            'adaptive_threshold_last_100_mean': float(np.mean(
                [value['adaptive_threshold'] for value in values[-100:]])),
        })

    return {
        'schema': 1,
        'seed': seed,
        'phase_length': phase_length,
        'residual_stream': 'absolute Gaussian scores; scale 1.0 then 2.0; shared by both intervals',
        'target_miscoverage': 0.1,
        'fixed_threshold': fixed_threshold,
        'calibrator': calibrator.summary(),
        'phases': phases,
        'scope': ('synthetic scalar scale-shift mechanism check only; no environment, policy, '
                  'task-return, conditional-coverage, or safety claim'),
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261111)
    parser.add_argument('--phase-length', type=int, default=800)
    parser.add_argument('--output', type=Path,
                        help='optional JSON receipt path; otherwise print to stdout')
    args = parser.parse_args(argv)
    result = run_audit(seed=args.seed, phase_length=args.phase_length)
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    rendered = json.dumps(result, indent=2, sort_keys=True) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
