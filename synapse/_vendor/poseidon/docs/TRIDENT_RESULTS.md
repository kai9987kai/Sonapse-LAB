# TRIDENT matched-seed pilot results

This report documents the replay-verified release receipt at `outputs/trident_experiments/RECEIPT.json`. It evaluates identity-gated directed residual control around the frozen Tidal core in the synthetic TidePool environment; it does not constitute evidence of general real-world capability or model promotion.

## Configuration and integrity

| Setting | Value |
|---|---|
| Profile | `poseidon-trident-experiment-v1` |
| Environment | TidePool `tidepool-v1` |
| Scarcity | 2.5 |
| Horizon | 64 steps per episode |
| Seeds | 88000000–88000003 |
| Arms | Core, TRIDENT, Erased, Shifted, Directed-support gate off, Residual gate off |
| TRIDENT decisions | 256 across four episodes |
| Receipt canonical SHA-256 | `333c309a43a447cf8eddabc61f008df7bc24e09011ec106b29bf44ee82c77c0e` |
| Receipt file SHA-256 | `5d91e7016c4ee10b3217249fb605a13f6a4b390804c653141b88847b8b3b6027` |
| Tidal checkpoint SHA-256 | `a053104ae63dc7f86baa15b4b34e46730e1ccb92117f221801fb21f90d2d6c27` |
| TidePool source SHA-256 | `f64b06b9434827b8ad6ce84212d8a7c62648abefa70661283de8b281c0acdbee` |

All six arms were evaluated synchronously on identical seeds. The receipt records each action, observation, transition reward, state delta, and final world snapshot. Verification reconstructs controllers and replays both controller policy decisions and TidePool environment transitions bit-for-bit. Verification reports `verified: true`; `promotion` is false.

## Descriptive outcomes

| Controller / Arm | Mean reward | Population SD | Survival | Mean steps |
|---|---:|---:|---:|---:|
| Frozen Tidal core | 6.4011 | 0.0781 | 100% (4/4) | 64.00 |
| TRIDENT (intact, gated) | 6.4272 | 0.0689 | 100% (4/4) | 64.00 |
| Erased (residual zeroed) | -0.2770 | 0.8882 | 0% (0/4) | 54.25 |
| Shifted (circular time-shift) | 1.6144 | 2.9479 | 50% (2/4) | 54.50 |
| Directed-support gate off (`no_topology`) | -0.4276 | 0.7661 | 0% (0/4) | 52.75 |
| Residual gate off (`ungated`) | -0.9396 | 0.1839 | 0% (0/4) | 47.00 |

### Paired performance versus TRIDENT

- **vs Frozen core:** +0.0262 mean reward delta (2 wins, 1 tie, 1 loss; 100% vs 100% survival).
- **vs Erased residual:** +6.7043 mean reward delta (4 wins, 0 ties, 0 losses; 100% vs 0% survival).
- **vs Shifted residual:** +4.8128 mean reward delta (4 wins, 0 ties, 0 losses; 100% vs 50% survival).
- **vs directed-support gate off:** +6.8548 mean reward delta (4 wins, 0 ties, 0 losses; 100% vs 0% survival).
- **vs residual gate off:** +7.3668 mean reward delta (4 wins, 0 ties, 0 losses; 100% vs 0% survival).

## TRIDENT telemetry

Across 256 decisions, TRIDENT executed 19 candidate residual overrides. All 19 overrides (100%) satisfied identity specificity (exhibiting positive advantage strictly against both erased and shifted counterfactual controls).
Mean telemetry values recorded in the receipt:
- Mean residual norm: 0.117037
- Mean intact advantage: 0.161878
- Mean erased advantage: 0.149396
- Mean shifted advantage: 0.151212
- Mean patches tracked: 3.988281
- Fallback reasons handled: `insufficient_intact_margin`, `identity_not_specific_vs_erased`, `identity_not_specific_vs_shifted`, `policy_preferred`.

## Reproduce and verify

```powershell
python -m poseidon trident-experiment --episodes 4 --max-steps 64 --scarcity 2.5 --seed 88000000
python -m poseidon verify-trident outputs/trident_experiments/RECEIPT.json
```

The seed range, active Tidal checkpoint, and TidePool environment source must match this receipt to reproduce its exact SHA-256 seal. TRIDENT does not modify neural weights or alter default baseline execution. See [TRIDENT architecture design](TRIDENT_DESIGN.md).
