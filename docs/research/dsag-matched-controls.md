# DSAG matched controls and evidence boundary

The core implementation is an engineering artifact. A smoke run, random initialization, or telemetry trace does not establish that DSAG improves task behavior. Promotion remains disabled until paired held-out trials exist.

## Preregistered minimum contrasts

| Arm | Configuration | Purpose |
|---|---|---|
| C0 | Legacy with DSAG absent | Exact host baseline |
| C1 | Single recurrent residual with matched parameters | Capacity control |
| C2 | Two independent recurrent residuals with matched compute | Two-processor control |
| C3 | DSAG with second soma ablated | Soma contribution |
| C4 | DSAG without along-axon inputs | Branch contribution |
| C5 | DSAG with scalar-only packets | Rich-message contribution |
| C6 | DSAG with frozen plasticity | Adaptation contribution |
| C7 | DSAG with neutral phase gate | Phase contribution |
| C8 | DSAG with fixed topology | Structural contribution |
| C9 | DSAG with shuffled branch assignment | Routing-information control |
| C10 | DSAG with decoupled somas | Coupling contribution |
| C11 | DSAG with packet loss / delay perturbation | Transport robustness |
| C12 | DSAG with parameter and FLOP matched ordinary recurrent block | Capacity and cost control |
| C13 | DSAG against two independent host neurons under equal resource limits | Primary independent-neuron contrast |

Freeze seeds, train/validation/test splits, host checkpoints, current limits, learning schedule, and cost budgets before trials. Report paired seed differences, confidence intervals, failures, latency, parameter count, state bytes, packet costs, and all ablations. The key planned contrasts are DSAG versus C2 and C13, not only DSAG versus C0.

No confirmatory study or task-level improvement claim is included in this implementation. The current executable smoke protocol is tagged `synthetic_fixture`; it is not biological evidence, safety evidence, or held-out performance evidence.
