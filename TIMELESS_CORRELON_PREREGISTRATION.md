# Timeless Correlon — Preregistration and Falsification Plan

**Version:** 1.0
**Research series:** Timeless Correlon
**Parent branch:** `agent/phase5c-interventional-representation-falsification`
**Working branch:** `agent/timeless-correlon-phase-t0-t3`

## Scope and continuity

This is a new research series. Phase 1–5C files and verdicts are not modified. In particular, Phase 5C remains **C — PLACEBO-EQUIVALENT / NO-GO** for a distinct interventional representation, and the surviving Phase-4 interpretation remains a descriptive persistent relation-mode detector.

The present program asks a narrower question:

> Can a relational state determine an internal flow or a preferred temporal parametrization without taking external physical time as an input?

The program does not claim to create change from nothing. It tests whether an already existing relational state and its allowed algebra can determine a generator, an internal flow, or a clock correspondence.

## Frozen hypotheses

| ID | Frozen hypothesis | Status before test | Falsification target |
|---|---|---|---|
| H_T0 | Equal-time covariance `C` alone uniquely determines temporal dynamics or physical time flow. | unresolved | Same-`C`, different-dynamics counterexample |
| H_T1 | In a Gaussian Gibbs system with known canonical structure, `C` determines the Hamiltonian generator up to one positive time scale. | positive control | Recovery error, vector-field alignment, sample-size monotonicity |
| H_T2 | A faithful state and observable algebra reproduce modular flow and agree with Gibbs Hamiltonian flow up to beta rescaling. | theorem reproduction control | Exact finite-dimensional implementation test |
| H_T3 | The compressed Phase-4 Correlon relation representation reconstructs the relevant thermal generator better than matched low-dimensional placebos. | unresolved | Paired held-out generator error and specificity gate |

## Anti-circularity firewall

1. T0 C-only inputs contain no time labels, lagged covariance, derivatives, trajectory order, true drift, or probability current.
2. Any lagged quantity is an evaluation diagnostic only; it is not an input to a timeless reconstruction.
3. A supplied canonical symplectic matrix `J`, Poisson bracket, or observable algebra is counted as supplied structure, not as an emergent result.
4. A global positive scale is treated as time-unit gauge freedom and is optimized only for evaluation.
5. Time orientation, duration, entropy production, and proper time are not identified with modular flow by default.
6. T2 is theorem/implementation calibration, not Correlon-specific evidence.
7. Equality of the Correlon whitened cross-covariance operator and CCA is tested explicitly. If they are numerically identical, no Correlon-specific novelty claim is allowed.

## Phase T0 — same covariance, different dynamics

Two-dimensional stationary Ornstein–Uhlenbeck systems are used:

```text
A0 = -gamma I
A1 = -gamma I + Omega J
J  = [[0, -1], [1, 0]]
D  = 2 gamma I
Sigma = I
```

Both systems have exactly the same stationary covariance `Sigma=I`. For `Omega>0`, the second system has a non-zero rotational probability current and different temporal dynamics.

Frozen grid:

- `gamma = {0.25, 0.5, 1.0}`
- `Omega = {0.0, 0.25, 0.5, 1.0, 2.0}`
- trajectory length `5000`, burn-in `1000`, `dt=0.01`
- 100 seeds (`0..99`)

The primary covariance gate uses the analytic stationary covariance equality, with pooled empirical covariance estimates reported as a finite-sample diagnostic. The dynamics gate uses the normalized theoretical generator difference for all `Omega>0` rows.

**Gate:** exact same covariance and 95% CI lower bound of normalized generator/current distinction at least `0.20`. If passed, the universal C-only hypothesis is frozen as falsified.

## Phase T1 — restricted Gaussian-Gibbs recovery

The canonical state is `z=(q,p)` with supplied canonical `J`, quadratic Hamiltonian `H(z)=0.5 z^T K z`, and Gibbs covariance `C=(beta K)^(-1)`. The reconstruction receives only an estimated covariance and uses `K_hat = inverse(C_hat)`, `A_hat = J K_hat`. The unknown global beta scale is evaluated after one positive scale calibration.

Frozen grid:

- dimensions `{2,4,8,16}`
- condition numbers `{1.5,3,10,30}`
- beta `{0.5,1,2,4}`
- sample sizes `{500,2000,10000}`
- 100 seeds per cell (`0..99`)

Positive-control gate:

- median scale-aligned generator error `<=0.10` for `N>=2000` and condition number `<=10`;
- median held-out vector-field cosine `>=0.95` in the same subset;
- median error improves as sample size increases.

The unknown-symplectic-structure ablation holds `C` fixed and compares the canonical `J` with an independently generated admissible skew complex structure. A successful ambiguity result is recorded as `C_REQUIRES_ADDITIONAL_ALGEBRAIC_STRUCTURE`.

## Phase T2 — modular-flow theorem reproduction

For a faithful Gibbs state `rho=exp(-beta H)/Tr(exp(-beta H))`, the modular Hamiltonian is `K_mod=-log(rho)=beta H + c I`. The implementation compares

```text
sigma_s(A) = exp(i s K_mod) A exp(-i s K_mod)
alpha_t(A) = exp(i t H) A exp(-i t H)
```

after one positive global scale fit on a frozen orthonormal Hermitian observable basis. The exact gate is generator error `<=1e-8` and flow error `<=1e-6` (with a relaxed numerical diagnostic at `1e-6` for generators).

Negative controls include unrelated non-Gibbs states, eigenvector-rotated Gibbs states, and a deliberately over-restricted diagonal observable algebra.

Passing T2 validates known modular-flow mathematics only. It does not show that Correlon generated time, nor does it establish a nontrivial state-independent outer modular flow.

## Phase T3 — Correlon relation-to-state bridge

The model is a coupled Gaussian Gibbs system with `X,Y in R^d`, `d in {4,8,16}`, low-rank cross-block coupling rank `{1,2,4}`, and coupling strength `{0.1,0.25,0.5,1.0}`. The full covariance is the oracle reference.

The frozen Correlon operator is the Phase-4 normalized whitened cross-covariance:

```text
Q = Cxx^(-1/2) Cxy Cyy^(-1/2)
```

The leading `k` singular modes reconstruct a rank-`k` cross block. The same externally supplied canonical `J` is applied to every method.

Methods:

- Correlon whitened cross-mode reconstruction;
- block-diagonal no-cross relation;
- PCA rank-matched covariance reconstruction;
- random rank-matched whitened cross mode;
- raw truncated SVD of `Cxy`;
- an explicit CCA-equivalent implementation identity check;
- full covariance oracle (not an admissible compressed baseline).

Training samples per case: `2000`; held-out samples: `1000`; eigenvalue floor: `1e-8`. Development seeds are `0..49`, validation seeds `1000..1049`, and confirmation seeds `10000..10199`.

Validation selects the Correlon rank and the best non-oracle low-dimensional baseline. Those selections and all numerical settings are written to `results/timeless_correlon_config.json`, hashed with SHA-256, and frozen before confirmation is opened.

Primary specificity gate:

```text
relative_gain = 1 - error_Correlon / error_best_baseline
CI95_low(relative_gain) > 0.10
```

If Correlon and CCA are numerically identical within `1e-10`, the verdict is `CORRELON_EQUALS_CCA_FOR_THIS_OPERATOR` regardless of the raw error ranking.

## Confirmation discipline

Development permits implementation and numerical-stability fixes only. Validation freezes the selection. Confirmation is run once, with no retuning, intermediate stopping, or threshold changes. An independent validator recomputes gates from raw CSV files using a separate bootstrap RNG.

## Forbidden conclusions

The following phrases are not permitted without a separate proof and experiment:

- “time has been derived from correlation”;
- “Correlon explains time”;
- “quantum gravity has been derived”;
- “spacetime emerges from Correlon”;
- “causality emerges from Correlon”.

## Reproducibility commands

```bash
python experiments/Timeless_T0_sameC_no_go.py
python experiments/Timeless_T1_gaussian_gibbs.py
python experiments/Timeless_T2_modular_flow.py
python experiments/Timeless_T3_relation_state_bridge.py --stage development
python experiments/Timeless_T3_relation_state_bridge.py --stage validation
python experiments/Timeless_T3_relation_state_bridge.py --stage confirmation
python experiments/Timeless_validate.py
python experiments/Timeless_make_report.py
```

The aggregate runner is `python experiments/Timeless_run_all.py`; it enforces the phase order and refuses confirmation if the validation freeze is absent.
