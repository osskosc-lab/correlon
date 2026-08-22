# Correlon Zero — Adversarial Invariance Falsification Preregistration

**Protocol:** `CORRELON_ZERO_ADVERSARIAL_INVARIANCE_FALSIFICATION_v1.0`  
**Preregistration version:** `1.0`  
**Status:** FROZEN BEFORE T2-T4 IMPLEMENTATION  
**Repository:** `osskosc-lab/correlon`  
**Working branch:** `agent/correlon-zero-adversarial-invariance`  
**Continuity base:** `a727f11e95e40d7b0919dc87dd9730bfc16abc69`  
**T0 audit commit:** `3d8dd92`  

## 1. Purpose and claim firewall

This experiment does not try to support Correlon. It attempts to falsify the frozen v1 observable against representation changes, mechanism destruction, matched conventional statistics, common causes, predictive baselines, and purpose-built false positives.

The strongest allowed positive statement is:

> The frozen scalar behaves as a transformation-selective relational invariant in the tested synthetic domain.

The experiment cannot establish direct causation, a particle or field, ontological reality, or implications for spacetime, gravity, consciousness, quantum mechanics, or cosmology. Simulation success is not physical discovery.

The Timeless T3 result that this whitened cross-covariance construction is CCA-equivalent remains active. A positive v1 result would not establish uniqueness unless v1 also exceeds every frozen conventional baseline by the frozen margin.

## 2. Frozen hypothesis

For paired time series `X in R^(T x d)` and `Y in R^(T x q)`, the frozen v1 scalar must retain its value under transformations that preserve a persistent directed cross-space mechanism and must lose its value when that mechanism is destroyed.

For a nonnegative scalar score `S`, define directional retention

```text
retention(S_original, S_transformed)
    = clip(S_transformed / max(S_original, 1e-12), 0, 1).
```

The scalar needs no coordinate alignment, so `align_g` is the identity. Score increases are capped at full retention; a destroy transform cannot be credited merely because it increases a false-positive score.

For each target world and metric:

```text
P_world = min retention over P1..P7
D_world = max retention over D1..D8
Delta_world = P_world - D_world
```

For each confirmatory seed, aggregate across the three target classes conservatively:

```text
P_seed = min_target_class P_world
D_seed = max_target_class D_world
Delta_seed = P_seed - D_seed
```

The primary research question is whether `Delta_seed` for Correlon v1 exceeds the frozen threshold and exceeds the per-seed strongest baseline.

## 3. Exactly one primary Correlon metric

### 3.1 Frozen Phase-4 operator

Each sliding window is centered separately. With `epsilon=1e-5`:

```text
Q_t = invsqrt(Cxx_t, epsilon) @ Cxy_t @ invsqrt(Cyy_t, epsilon)
Q_t = U_t diag(s_t) V_t^T
```

`invsqrt(C,epsilon)` uses the Phase-4 formula `V diag(1/sqrt(max(eigenvalue,0)+epsilon)) V^T`.

Frozen window settings:

- `window=140`
- `step=40`
- `max_lag=3`
- `matched_null_replicates=6`
- circular shifts sampled uniformly as integers in `[T/4, 3T/4)`
- the same deterministic null-RNG stream is used for the original and transformed copy of a seed

For each window sequence:

```text
gap = median(s1_t - s2_t)
joint_t = sqrt((u_t dot u_(t+1))^2 * (v_t dot v_(t+1))^2)
floor = quantile_0.10(joint_t)
T_iso = gap_observed - median(gap_circular_shift_null)
T_floor = floor_observed - median(floor_circular_shift_null)
```

### 3.2 Frozen scalar aggregation

The only primary Correlon score is:

```text
CorrelonZero_v1 = sqrt(clip(T_iso,0,1) * clip(T_floor,0,1)).
```

The geometric mean requires both spectral isolation and lower-tail mode continuity. `T_pers`, raw strength, relative gap, mean persistence, and minimum persistence remain diagnostics and cannot replace or reweight the primary score after this freeze.

### 3.3 Frozen implementation identity

The source definition is anchored to the unmodified Phase-4 implementation:

- source file: `experiments/Phase4_correlon_core_persistence.py`
- Git blob: `bf80ad90beecbbcc8c3638e3250e57a2b56eded4`
- SHA-256: `faa6e602fe507eba37fb902ff191a0bcf348b225766ff4ed676e37438e8fc8cb`

The Correlon Zero implementation must be a tested transcription of these operator/readout equations plus the frozen scalar aggregation above. Before confirmatory execution, a machine-readable manifest will hash the preregistration, config, metric module, generator module, transform module, baseline module, analysis module, adversary module, and CLI. Recording that implementation hash may not alter any semantic rule.

## 4. Data, preprocessing, and deterministic seed policy

Default paired-world shape:

- `T=600`
- `d=q=6`
- burn-in `200`
- independent Gaussian innovations unless a null definition explicitly supplies a latent common input

Metric preprocessing is limited to window-wise centering already present in Phase 4. No global rank selection, target-specific normalization, denoising, outlier deletion, seed exclusion, or post-confirmatory rescaling is allowed.

Seed blocks:

- pilot: `0..19`
- confirmatory: `10000..10199`
- adversarial search RNG: `424242`
- bootstrap RNG: `99173`

Every generator class and transformation receives a stable integer namespace defined in the committed config. Python's randomized `hash()` is forbidden. Failed seeds remain recorded. A seed may be excluded only for a thrown implementation/numerical error, which triggers `INCONCLUSIVE_IMPLEMENTATION` rather than silent deletion.

## 5. Frozen target mechanism and worlds

The target mechanism is a fixed-orientation directed cross-lag edge active throughout the record. It may be bidirectional, nonlinear, or history-dependent, but it must not be an exogenous common cause.

### Target classes

1. `persistent_shared_mode`: stable bidirectional rank-1 VAR relation. Within-space AR coefficient `0.65`; cross coefficients `0.28` in each direction; fixed independently drawn orthonormal loading directions; innovation scale `0.60`.
2. `direct_coupling_SCM`: `X` AR coefficient `0.65`; `Y` AR coefficient `0.55`; fixed delay `2`; nonlinear edge `0.65 * v * tanh(u^T X_(t-2))`; innovation scale `0.60`.
3. `history_dependent_relational_mode`: `X` AR coefficient `0.65`; exponential memory coefficient `0.92`; memory injection `0.08*u^T X`; `Y` AR coefficient `0.55`; nonlinear history edge `0.90*v*tanh(memory)`; innovation scale `0.60`.

All stability checks are deterministic. An unstable coefficient realization is an implementation failure, not a seed to replace.

### Negative classes

1. `independent_noise`: independent AR channels, no shared innovations or cross edges.
2. `common_driver`: latent AR(1) factor (`rho=0.93`) drives both spaces through fixed loadings; no `X<->Y` edge.
3. `matched_low_rank`: contemporaneous iid rank-1 factor shared across spaces with matched marginal scale; no temporal or directed edge.
4. `matched_spectrum`: independently phase-randomized copies of a target realization, preserving each channel amplitude spectrum.
5. `matched_autocorrelation`: independently simulated per-space VAR(1) approximations with no cross-space coefficients.
6. `switching_mode`: direct coupling direction switches at the midpoint, so no persistent full-record identity exists.
7. `transient_mode`: direct edge is active only in the middle third.
8. `nonstationary_drift`: matched smooth deterministic drifts plus independent stochastic residuals; no direct edge.
9. `finite_sample_spurious`: the highest v1 score among eight preregistered independent high-autocorrelation candidates; multiple-testing selection is part of the null definition.
10. `mixture_without_relational_mechanism`: blocks from common-factor nulls with independently changing loadings; no directed cross edge.

Where feasible, generated outputs are rescaled to common channel variance without changing cross-edge status. Dimensionality and nominal sample length remain fixed for all confirmatory classes.

## 6. Frozen PRESERVE transformations

All random choices use transform-specific deterministic seeds.

| ID | implementation | ground-truth requirement |
|---|---|---|
| P1 node permutation | independent fixed permutations within X and Y | directed edge is relabeled, not removed |
| P2 channel scaling | positive log-uniform factors in `[0.5,2.0]` | invertible observation map |
| P3 affine offset | channel offsets uniform in `[-2,2]` channel SD | removed by frozen centering |
| P4 orthogonal basis rotation | independent Haar-QR orthogonal matrices for X and Y | invertible basis change |
| P5 observation noise | independent noise at `0.10` times channel SD | same latent/direct mechanism remains |
| P6 monotone time reparameterization | interpolation through `t -> t^1.25`, same output length and order | event order is strictly preserved |
| P7 coarse graining | non-overlapping mean over factor `2` | target class and direction persist at coarser resolution |

P1-P7 cannot be moved to DESTROY. The weakest preservation transform is always reported.

## 7. Frozen DESTROY transformations

| ID | implementation | destruction target |
|---|---|---|
| D1 temporal shuffle | independent full row permutations for X and Y | destroys temporal and paired cross-lag ordering while preserving marginals |
| D2 phase randomization | independent Fourier phases per channel | preserves amplitude spectrum, destroys cross-channel phase organization |
| D3 IAAFT surrogate | independent per-channel IAAFT, `30` iterations | preserves marginal ranks and approximately spectrum, destroys cross organization |
| D4 edge cutting | rerun the same target generator and seed with every cross coefficient set to zero | removes the structural directed edge |
| D5 source swap | X from the original, Y from an independent same-class seed; restore Y mean/SD | removes the paired mechanism |
| D6 common-driver replacement | fit a rank-1 common-factor null to top cross-covariance scale and within-channel persistence | replaces direct mechanism with exogenous common cause |
| D7 history destruction | fit and simulate X-only and Y-only VAR(1) models with independent innovations | preserves short-lag within-space behavior, removes cross history |
| D8 block shuffle | block length `25`, independent block orders for X and Y | preserves within-block local dynamics, destroys cross-block and paired ordering |

D1-D8 cannot be removed or reclassified after results. Ground-truth tests must verify their intended destruction before pilot execution.

## 8. Frozen conventional baselines

Every applicable baseline uses the same original and transformed datasets and the same retention/P/D aggregation.

1. `pearson`: RMS of the cross-space Pearson correlation matrix.
2. `mutual_information`: multivariate Gaussian mutual information from regularized covariance log-determinants, mapped to `[0,1)` by `1-exp(-MI/min(d,q))`.
3. `largest_covariance_eigenvalue`: largest eigenvalue of the centered joint covariance.
4. `pca_rank1_explained_variance`: largest joint covariance eigenvalue divided by total eigenvalue sum.
5. `spectral_coherence`: mean magnitude-squared coherence of the first within-space principal-component time series, Welch `nperseg=128`.
6. `autocorrelation_persistence`: mean absolute lag-1 autocorrelation across all channels.
7. `linear_CKA`: centered linear CKA between X and Y.
8. `VAR_ARX_predictive`: symmetric mean cross-predictive gain, comparing within-space AR to ARX in both directions with lags `1..3` and ridge `1e-3`.

CCA is not reported as a separate numerical baseline because the T0 audit establishes exact operator identity for the relevant whitened SVD; this identity is stated in every interpretation table. Linear CKA remains the required representation-similarity baseline.

No baseline may be removed because it outperforms v1.

## 9. Pilot-only threshold calibration

The preserve/destroy thresholds are already on a bounded scale and are frozen now:

- minimum mean `Delta_C = 0.10`
- minimum mean paired baseline advantage `=0.10`
- maximum false-positive rate `=0.05`
- representation retention tolerance for P1/P4 `=0.90`
- maximum P1/P4 failure rate `=0.05`
- lower-tail rule: 5th percentile of `Delta_seed` must be `>0`

The only scale-dependent threshold is each metric's positive-call threshold. It will be computed mechanically from the 20 pilot `independent_noise` scores:

```text
tau_metric = max(empirical_quantile_0.95(pilot independent_noise score), 1e-6)
```

The quantile method is NumPy `method="linear"`. Threshold values are written to the frozen config and committed before confirmation. All negative classes other than `independent_noise` are prohibited from threshold calibration because including hard nulls would hide false positives.

Pilot results cannot change the metric, generator parameters, transform membership, baseline definitions, support criteria, or seed blocks. Code/numerical defects may be fixed and require a complete pilot rerun; semantic changes require Correlon Zero v2 and a new preregistration.

## 10. Confirmatory statistics

The confirmatory unit is one seed after conservative aggregation across the three target classes.

For every metric report:

- mean, median, standard deviation
- 95% percentile bootstrap CI of the mean (`5000` paired resamples)
- 5th percentile
- all per-seed values
- count with `Delta_seed <= 0`
- weakest preserve transform and strongest residual destroy transform

Baseline advantage is computed per seed as:

```text
advantage_seed = Delta_Correlon_seed - max_baseline(Delta_baseline_seed).
```

The maximum is taken within each seed, so no favorable single baseline selection is possible.

False-positive rate is the fraction of confirmatory negative worlds with `score > tau_metric`, reported by negative class and pooled. Common-driver and matched-statistic classes are never pooled away in the decision.

## 11. Anti-Correlon Generator

The adversary is a preregistered deterministic random search of `300` null candidates. Every candidate has `direct_XY_edge=false` by construction and is generated only from latent common factors, independent residual dynamics, deterministic drift, or their mixtures.

Search space:

- latent rank `{1,2,3}`
- latent AR coefficient `[0,0.995]`
- latent oscillator frequency `[0,0.45]` cycles/sample
- latent strength `[0.2,3.0]`
- independent noise scale `[0.1,1.5]`
- drift strength `[0,2.0]`
- X/Y latent delays `{0,1,2,3,5,8}`
- sample length `{300,450,600,750}`
- fixed output dimensions `6+6`

Objective: maximize the frozen raw `CorrelonZero_v1` score. The search cannot alter v1 or `tau_C`.

Formal adversarial falsification requires at least 3 distinct candidates with `score > tau_C`. Maximum score, crossing count/rate, trajectory, top cases, and the best example are reported even if the formal repeated-crossing rule is not met.

## 12. Decision rule and label precedence

Support requires all of:

1. mean `Delta_C >=0.10`;
2. 5th percentile `Delta_C >0`;
3. pooled and every named negative-class FPR `<=0.05`;
4. mean paired baseline advantage `>=0.10` and its bootstrap lower bound `>0`;
5. P1/P4 failure rate `<=0.05`;
6. fewer than 3 adversarial positive nulls;
7. no common-driver or matched-statistic class explains the positive boundary.

All criteria and all secondary failure labels are recorded. A single primary label is assigned using this precedence:

1. `INCONCLUSIVE_IMPLEMENTATION` if ground-truth tests, numerical stability, or frozen implementation reproduction fails.
2. `FALSIFIED_REPRESENTATION_DEPENDENCE` if P1/P4 failure rate exceeds `0.05`.
3. `FALSIFIED_COMMON_DRIVER` if common-driver FPR exceeds `0.05`.
4. `FALSIFIED_MATCHED_NULL` if matched-low-rank, matched-spectrum, or matched-autocorrelation FPR exceeds `0.05`.
5. `FALSIFIED_BY_BASELINE` if the strongest baseline equals/exceeds mean Correlon separation.
6. `FALSIFIED_ADVERSARIAL` if at least three admissible adversarial nulls cross `tau_C`.
7. `SURVIVES_WITHOUT_UNIQUENESS` if preserve/destroy and FPR criteria pass and Correlon remains above the strongest baseline, but the `0.10` uniqueness margin fails.
8. `SURVIVES_ZERO` only if every support criterion passes.
9. `INCONCLUSIVE_POWER` only if implementation passes but required precision cannot be obtained from the frozen 200 seeds.

If several scientific failures occur, the precedence chooses the primary label without hiding the remaining failures.

## 13. Stop discipline

An immediate implementation stop occurs before scientific runs if preserve ground-truth tests fail, destroy tests fail their stated oracle target, the Phase-4 equations cannot be reproduced, or numerical defects invalidate v1.

A scientific failure discovered in the fixed v1 battery permanently stops metric rescue or v2 theory work in this run. It does **not** suppress the remaining already-preregistered v1 confirmatory transforms or the Anti-Correlon search, because those outputs are required to identify the full falsification surface. No v2 metric will be proposed or evaluated on the confirmation seeds.

## 14. Frozen outputs and commands

Required implementation paths and machine-readable outputs follow the user-specified layout. Additional detailed files may be added, but required files cannot be replaced.

Planned CLI:

```text
python -m experiments.correlon_zero.run_correlon_zero --config experiments/correlon_zero/config.json --stage test
python -m experiments.correlon_zero.run_correlon_zero --config experiments/correlon_zero/config.json --stage pilot
python -m experiments.correlon_zero.run_correlon_zero --config experiments/correlon_zero/config.json --stage confirmatory
python -m experiments.correlon_zero.run_correlon_zero --config experiments/correlon_zero/config.json --stage adversarial
python -m experiments.correlon_zero.run_correlon_zero --config experiments/correlon_zero/config.json --stage report
```

The exact one-command rerun will be frozen in the final report. Raw CSV and JSON outputs retain configuration and implementation hashes.
