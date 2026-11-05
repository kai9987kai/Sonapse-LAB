# Graft validation continuation

**User instruction:** continue the approved research-led adaptive-graft work.

**Scope:** repair a demonstrated age-counter boundary bug and strengthen the
already-approved evidence tools. Keep the neural algorithm, learning objective,
teacher fixture, candidate graph, growth settings and existing live run intact.
No new architecture, data import, publication or live restart is included.

- [x] Bound `min_age` to signed-int64 representability and saturate active ages
  at its maximum; reproduce overflow before repair and verify state restoration.
- [x] Add reporter-only target distances, planar path length and clipped action
  statistics to frozen/train/reference episodes and paired diagnostics.
- [x] Add full-trajectory AUC, final-window summaries, isolated pre/post structural
  event error and old-target observations to the supervised probe. Keep its
  optimization objective and random streams unchanged.
- [x] Freeze the confirmation protocol before observing seeds 1000003–1000019.
  Primary endpoint: paired fixed-minus-adaptive normalized recovery AUC.
- [x] Run a bounded 640-step, seven-control descriptive RL study for forage and
  balanced go/no-go, including behavior diagnostics and post-maturation updates.
- [x] Independently review changes, run the complete suite and source checks,
  and report all outcomes, including null/negative results and limits.
- [x] Preserve original state hashes and process4272/port8770 without commands.

## Audit evidence driving this continuation

The prior pilot's zero return change concealed parameter/action changes.
Its go/no-go reward is constant outside the approach threshold. The run also
ended before any new age-20 record matured. The supervised endpoint at update80
coincides with a structural event, motivating trajectory metrics.

A valid `min_age=2**63-1` immediately overflowed active ages on increment, and
`min_age=2**63` was accepted despite unrepresentable state. Three regressions
were observed failing before the bounded/saturating repair.

Before freeze, independent review found short unit probes had used seeds3,4,5.
The confirmation namespace was therefore offset by1000000 before any declared
80+80 confirmation run. The seed count, algorithm, budgets and analysis stay
fixed; this avoids overlap rather than selecting by outcomes.
