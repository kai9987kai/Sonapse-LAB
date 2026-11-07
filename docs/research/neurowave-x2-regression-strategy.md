# Sonapse NeuroWave-X2 — Regression Prevention & Checkpoint Migration Strategy

## 1. Zero-Gain Identity Guarantee

The primary architectural requirement of Sonapse NeuroWave-X2 is absolute non-regression across all existing Sonapse-LAB modules, connectomes, and training benchmarks.

### Formal Proof of Zero-Gain Identity
Let the base recurrent connectome dynamics be:
$$h_{t+1}^{\text{base}} = \tanh(W_{\text{base}} h_t + W_{\text{graft}} r_t + b)$$

Under NeuroWave-X2, an auxiliary communication term $\Delta_{\text{comm}}(t)$ is injected scaled by $\gamma \in [0.0, 5.0]$:
$$h_{t+1} = \tanh(W_{\text{base}} h_t + W_{\text{graft}} r_t + b + \gamma \cdot \text{clip}(\Delta_{\text{comm}}(t), -c, c))$$

When `neurowave_x2_gamma = 0.0` or `neurowave_x2_enabled = False`:
$$\gamma \cdot \text{clip}(\Delta_{\text{comm}}(t), -c, c) = 0.0 \cdot \dots \equiv 0$$
Therefore:
$$h_{t+1} \equiv h_{t+1}^{\text{base}}$$
This holds **bit-for-bit** in IEEE 754 floating point arithmetic.

### Empirical Bit-for-Bit Verification
In `tests/test_neurowave_x2_regression.py::test_zero_gain_identity_guarantee`, identical inputs are passed to a baseline agent and a NeuroWave-X2 configured agent (`gamma=0.0`). The resulting action tensors are confirmed identical:
$$\| a_{\text{base}} - a_{\text{neurowave}(\gamma=0.0)} \|_\infty = 0.0$$

---

## 2. Checkpoint Migration Policy

### 2.1 Backward Compatibility
When loading a legacy checkpoint created before the introduction of NeuroWave-X2:
- The missing `neurowave_x2` dictionary in `checkpoint['state']` is detected cleanly.
- `IntegratedAgent.import_state()` initializes NeuroWave-X2 to its dormant default state (`enabled=False`, `step_count=0`).
- No exceptions are raised, and the model behaves identically to the pre-change code.

### 2.2 Forward Checkpoint Serialization
Modern checkpoints export a dedicated `neurowave_x2` section containing:
```json
{
  "basis_version": 1,
  "config": {
    "enabled": false,
    "residual_gain": 0.0,
    "protocol": "adaptive",
    "spectral_basis": "hybrid"
  },
  "topology_routes": [],
  "step_count": 0,
  "model_state": { ... }
}
```
All state is strictly versioned and validated upon deserialization.
