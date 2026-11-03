# Correlon Phase 5C — Nonlinear / Nonstationary Interventional Representation Falsification

## Final verdict

**C — PLACEBO-EQUIVALENT / NO-GO**

Phase 5C does not support Correlon as a distinct interventional representation. The Correlon mixture produced a small positive confirmation gain, but the gain was far below the preregistered 10% requirement and was significantly worse than the strongest frozen placebo (PCA).

Under the preregistered stopping rule, the current Correlon causal/interventional extension is terminated. Correlon remains viable only as a descriptive persistent relation-mode detector under the tested program.

## Approximate Twin Gate

All observational-equivalence controls passed before confirmation was opened:

| Check | Result | Threshold | Verdict |
|---|---:|---:|---|
| Direct/Common classifier AUC | 0.5007; 95% CI [0.4973, 0.5051] | upper <= 0.60 | PASS |
| Median normalized Y RMS discrepancy | 0.02433 | <= 0.10 | PASS |
| Max abs Correlon standardized difference | 0.00777 | <= 0.20 | PASS |

## Validation freeze

The strongest baseline was **exponentially weighted nonlinear regression** with validation BIRE `0.802431`.

Frozen representation choices:

- `B* = ew_nonlinear`
- `w_C* = 0.25`
- strongest placebo `P* = PCA`
- `w_P* = 0.25`

At Validation, Correlon gain was `1.389%`; PCA gain was `1.871%`.

## Confirmation — 200 seeds

| Quantity | Result |
|---|---:|
| Baseline BIRE | 0.786393 |
| B* + Correlon BIRE | 0.775145 |
| B* + PCA BIRE | 0.771622 |
| Correlon gain G_C | 1.4303% |
| 95% CI for G_C | [1.1782%, 1.7002%] |
| PCA gain G_P | 1.8784% |
| G_C - G_P | -0.4481 percentage points |
| 95% CI for G_C - G_P | [-0.5691, -0.3286] percentage points |

Safety controls:

- Common-driver error delta: `+0.00733`, 95% CI `[0.00629, 0.00841]` — PASS (`upper <= 0.02`).
- Sham error delta: `+0.00607`, 95% CI `[0.00500, 0.00725]` — PASS (`upper <= 0.02`).

## Gate decision

| Gate | Requirement | Result |
|---|---|---|
| Gate 1 | Approximate Twin | PASS |
| Gate 2 | `CI95_low(G_C) > 10%` | **FAIL** |
| Gate 3 | Common-driver harm upper <= 0.02 | PASS |
| Gate 4 | Sham harm upper <= 0.02 | PASS |
| Gate 5 | `CI95_low(G_C-G_P*) > 5 pp` | **FAIL** |

## Scientific interpretation

The experiment rejects the strong Phase-5C claim that the frozen Correlon mode supplies a materially and specifically useful interventional coordinate beyond strong nonlinear/nonstationary baselines.

The positive `~1.43%` gain is not zero, but it cannot be promoted to evidence for Correlon-specific causal-response utility because:

1. the preregistered minimum gain was 10%, and the full 95% CI remains near 1–2%;
2. PCA, selected and frozen as the strongest placebo during Validation, improves BIRE more (`~1.88%`);
3. the specificity contrast is negative with a fully negative 95% CI.

The result is therefore best classified as **placebo-equivalent**, not as a weak Correlon-specific signal.

## Independent validator

A separate validator re-read frozen selections and all 200 confirmation seed-level error files and recomputed the primary decision with an independent bootstrap RNG. It reproduced:

- baseline selection;
- Correlon and placebo weights;
- 200-seed count;
- point gains and safety deltas;
- Gate 1–5 pattern;
- final `C_PLACEBO_EQUIVALENT_NO_GO` verdict.

Independent-bootstrap Correlon gain CI was `[1.1641%, 1.6957%]`, consistent with the primary analysis.

## Stop rule consequence

Per preregistration:

**Phase 5C FAIL => end the current Correlon causal/interventional extension.**

The surviving Phase-4-safe definition remains:

> Correlon is a statistically isolated and temporally continuous dominant normalized cross-space relation mode, relative to a structure-preserving matched null.

No direct-causation or distinct interventional-utility claim is retained from Phase 5C.
