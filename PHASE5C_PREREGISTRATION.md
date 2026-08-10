# Correlon Phase 5C — Preregistration Freeze

## Status

Phase 5C is a new falsification experiment. It is not a rescue analysis of Phase 5 and does not re-open the falsified implication `Correlon-positive => direct causation`.

## Central hypothesis

**H5C:** A persistent relation mode extracted by the frozen Phase-4 Correlon operator is a useful low-dimensional representation for predicting held-out interventional responses in nonlinear, slowly nonstationary systems.

The scientific bridge is therefore:

`persistent relation mode -> ? -> useful coordinate for causal-response learning`

not causal identification.

## Frozen operator and readout

The Correlon operator remains the Phase-4 whitened cross-covariance SVD:

`Q_t = Cxx^(-1/2) Cxy Cyy^(-1/2)`.

The leading singular X-side mode defines the intervention coordinate `c = u_C^T deltaX`. The Correlon readout is RBF kernel ridge regression. No operator tuning is allowed after Validation begins.

## SCM family

- `X_t, Y_t in R^8`, latent scalar `Z_t`.
- Direct SCM: nonlinear delayed `X -> Y` response with slowly drifting true relation directions and modulated coupling strength.
- Common-driver SCM: `Z -> X` and `Z -> Y`, tuned for approximate observational equivalence to the Direct SCM.
- Nonlinearity: `f(s)=tanh(1.15 s)+0.42 s^3/(1+s^2)`.
- Slow drift is present in both X-side and Y-side relation directions and in coupling amplitude.

## Approximate Twin Gate

All must pass before confirmation seeds are opened:

1. Direct/Common classifier `AUC_95%_upper <= 0.60`.
2. Median normalized Direct/Common Y RMS discrepancy `<= 0.10`.
3. Maximum absolute standardized difference across frozen Correlon statistics `<= 0.20`.

Gate failure implies STOP.

## Interventions

Calibration uses directions `d1,d2` and amplitudes `[-1.0,-0.5,+0.5,+1.0]` (8 active interventions).

Held-out testing uses unseen directions `d3..d8` and amplitudes `[-1.5,-0.75,+0.75,+1.5]` (24 active interventions), plus 8 sham interventions. Intervention estimates include Gaussian measurement noise with four replicates.

## Baselines

Validation selects the single best baseline by BIRE from:

- Rank-1 ARX
- Volterra Rank-1
- Nonlinear reduced-rank regression
- Local nonlinear reduced-rank regression
- Exponentially weighted nonlinear regression
- Full-vector RBF kernel ridge
- Nonlinear state-space model

Placebo representations are PCA, circular-shift and random mode.

## Mixture rule

`tau_hat = (1-w) tau_hat_B* + w tau_hat_R`, with `w in {0,0.25,0.50,0.75,1.0}`. Baseline, Correlon weight, strongest placebo and placebo weight are selected on Validation only and frozen for Confirmation.

## Split

- Development: 50 seeds (`0..49`)
- Validation: 50 seeds (`1000..1049`)
- Confirmation: 200 seeds (`10000..10199`)

Confirmation is opened once, with no retuning and no intermediate stopping.

## Primary endpoint and gates

Primary endpoint: BIRE.

Correlon gain: `G_C = 1 - BIRE_(B*+C)/BIRE_B*`.

Strong GO requires every gate:

1. Approximate Twin Gate passes.
2. `CI95_low(G_C) > 0.10`.
3. Common-driver harm: `CI95_upper(E_common^(B+C)-E_common^B) <= 0.02`.
4. Sham noninferiority: `CI95_upper(E_sham^(B+C)-E_sham^B) <= 0.02`.
5. Correlon specificity: `CI95_low(G_C-G_P*) > 0.05`.

If Phase 5C fails, the current `relation mode -> causal/interventional utility` research program stops. The surviving interpretation is Correlon as a descriptive persistent relation-mode detector.

## Development engineering freeze

Before Validation was rerun from the beginning, Development exposed unnecessary repeated fitting cost. A non-semantic cache/refactor and a fixed recent-tail implementation for nonlinear reduced-rank regression were frozen. No gate thresholds, seed blocks, intervention sets, candidate weights, endpoints, or confirmation rules were changed after this freeze.

Frozen configuration version: `Phase5C-preregistered-v1.1-dev-freeze`.

Frozen configuration SHA-256: `71b546652dc41fe2eca419b69534ac44a68a96e482461d1e4cf163612f830e39`.
