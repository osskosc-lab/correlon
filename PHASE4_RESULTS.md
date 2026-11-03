# Correlon Phase 4 — Core Existence & Persistence Falsification

## Question
Can the SRRD-independent Correlon Core distinguish a genuinely persistent isolated rank-1 cross-space mode from finite-sample/null structure, degenerate rank-2 structure, switching modes, transient bursts, and a pure common-driver correlation?

The operator is

\[
Q_t=(\Sigma_{xx,t}+\epsilon I)^{-1/2}\Sigma_{xy,t}(\Sigma_{yy,t}+\epsilon I)^{-1/2},
\]

with dominant SVD mode and projection-sign-invariant persistence.

## Design
- 6 scenarios
- 100 seeds each
- T=900, d=q=8
- sliding window 140, step 40
- matched circular-shift null preserving each space's internal autocorrelation/spectrum
- primary isolation statistic: `T_iso = median gap(obs) - median gap(null)`
- initial persistence statistic: mean lagged projection overlap above matched null
- diagnostic revision: lower-tail continuity (`q10` and minimum consecutive projection overlap)

## Phase 4A — initial mean-persistence test

Persistent rank-1 was clearly separated from the matched null:

- rank1 persistent: `T_iso = 0.1061 [0.1007, 0.1114]`, `T_pers = 0.4326 [0.4197, 0.4454]`
- null: `T_iso ≈ 0`, `T_pers = 0.0067 [-0.0019, 0.0154]`

However the *mean persistence* criterion was falsified as a sufficient discriminator:

- rank1 switching: `T_pers = 0.3593`
- rank2 degenerate: `T_pers = 0.2587`
- transient burst: `T_pers = 0.1429`

Thus a time-averaged persistence score can hide mode replacement, degeneracy, or temporary activation.

## Phase 4B — continuity-floor stress test

The diagnostic was tightened without changing the Correlon operator. We measured the lower tail of consecutive two-sided projection overlap.

Mean minimum consecutive overlap:

- rank1 persistent: `0.3796`
- common driver: `0.3771`
- rank1 switching: `0.1489`
- rank2 degenerate: `0.0569`
- transient burst: `0.0261`
- null: `0.0122`

This substantially improves distinction between one persistent isolated mode and switching / degenerate / transient structures.

## Strongest falsification result

The common-driver model is intentionally non-causal between X and Y: both are driven by an external latent Z. Nevertheless its statistics are essentially indistinguishable from the direct persistent rank-1 relation:

- common driver `T_iso = 0.1078 [0.1022, 0.1134]`
- rank1 persistent `T_iso = 0.1060 [0.1005, 0.1114]`
- common driver continuity floor `T_floor = 0.5344 [0.5012, 0.5676]`
- rank1 persistent continuity floor `T_floor = 0.5607 [0.5300, 0.5914]`

Therefore Phase 4 directly falsifies the interpretation

> persistent isolated Correlon => direct causal coupling.

The Correlon Core detects a persistent *relation mode*, not causal direction or causal mechanism.

## CORE-X verdict

### Supported
1. A persistent rank-1 relation produces a strong isolated dominant cross-space mode above matched circular-shift null.
2. Spectral gap is necessary: the equal-strength rank-2 case has high overall cross-correlation but a much weaker isolation statistic.
3. Lower-tail / continuity-floor persistence is more informative than mean persistence for detecting replacement and intermittency.

### Falsified / contracted
1. Mean persistence alone is not sufficient for a unique persistent Correlon.
2. Correlon detection alone cannot distinguish direct coupling from common cause.
3. Correlon must not be interpreted as causal, self-like, conscious, or particle-like without an additional experiment.

## Updated minimal definition

A Phase-4-safe definition is:

> Correlon = a statistically isolated and temporally continuous dominant normalized cross-space relation mode, relative to a structure-preserving matched null.

Causality is explicitly excluded from the definition.

## Next experiment — Phase 5

Phase 5 should test *interventional identifiability*:

- direct X→Y coupling versus latent common driver Z→{X,Y};
- matched observational covariance before intervention;
- unseen interventions on X only;
- criterion: whether Correlon-conditioned dynamics predict the post-intervention Y response beyond an observational baseline.

If direct coupling and common-driver cases remain indistinguishable under intervention, Correlon has no causal content and must remain a purely descriptive relation-mode theory.
