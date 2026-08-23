# Timeless Correlator T8 - Preregistration

**Version:** 1.0  
**Subtitle:** Non-Markov Memory Necessity and Standard-Theory Sufficiency Test  
**Source commit:** `f1e0227d44ef50e489c6a32cfc0941dbf5f6925e`  
**Branch:** `agent/timeless-correlator-t8-nonmarkov-memory`

## 1. Frozen scope and claim firewall

T0-T7 code, results, and verdicts are immutable. T8 asks whether fixed finite Markov
compression becomes inefficient for synthetic long-memory families and, if so, whether
standard fractional, memory-kernel, Volterra, predictive-state, or reservoir methods
are sufficient. T8 cannot establish Correlon, time emergence, new physics, or a
fundamental non-Markov ontology.

Because every finite record can be approximated by a sufficiently large finite model,
the strongest allowed Markov conclusion is compression inefficiency at the tested
resolution and budget.

## 2. Frozen generators

Positive controls are a finite AR(3) predictor, a four-dimensional linear Gaussian
state-space kernel, and a sum of three exponentials admitting finite state augmentation.

Targets are power-law kernels (`alpha` 0.25, 0.50, 0.75, 1.25), fractional-difference
kernels, a generalized-Langevin oscillatory power-law kernel, ARFIMA-like kernels
(`d` 0.10, 0.20, 0.35, 0.45), a nonlinear power-law Volterra response, and a
history-dependent intervention response.

Hard nulls are large finite AR, high-dimensional finite SSM, many-exponential kernels,
slow common drivers, nonstationary drift, regime switching, colored noise, and
measurement filtering. Ground-truth tags are isolated from detector inputs.

## 3. Information budgets

- B0: current observed state only.
- B1: one fixed finite lag window.
- B2: observed history up to `L`.
- B3: the same history plus an intervention specification.

Candidate and baseline models are compared only within the same budget, split, noise
realization, intervention set, and computational accounting.

## 4. Primary scaling test

History lengths are `16, 32, 64, 128, 256, 512, 1024`. Finite lag candidates are
`1, 2, 4, 8, 16, 32, 64, 128, 256`; state dimensions are
`1, 2, 4, 8, 16, 32, 64, 128`.

`p_star(L)` is the smallest lag whose omitted-kernel energy gives normalized prediction
error at most 0.05. If the grid is exhausted, the result is recorded as a lower bound,
not silently extrapolated. `d_star(L)` is the smallest tested exponential-state or
predictive-state dimension reaching the same 0.05 criterion. Parameter count, held-out
prediction NMSE, held-out intervention NMSE, long-horizon error, log likelihood, and
kernel reconstruction error are recorded per seed.

Constant, logarithmic, power-law, and linear scaling curves are fitted. Compression
inefficiency requires all of the following: at least one designated target has
nondecreasing complexity, final complexity at least four times initial complexity,
constant-model delta AIC at least 10 against the best increasing model, and the result
persists at an unseen larger history length. Hitting the grid ceiling is reported as a
budget lower bound and supports inefficiency only, never absolute non-Markovity.

Positive controls must saturate: AR(3) at lag no larger than 4, the SSM at state
dimension no larger than 4, and the three-exponential kernel at dimension no larger
than 4. Failure yields `COMPRESSION_TEST_INVALID` and stops the scientific run.

## 5. History-conditioned intervention gate

The critical pair matches current state, intervention amplitude, and noise, while
swapping only the prior history. Training interventions are pulse amplitudes 0.5 and
1.0 and step amplitude 1.0. Held-out interventions are pulse amplitudes 0.25 and 1.5
and an unseen paired-pulse separation.

The target gate is median `Delta_H_R >= 0.15` with bootstrap 95% lower bound above
0.10. The finite-Markov sufficient-state negative control must have median
`Delta_H_R <= 0.03`. A nonzero target effect shows insufficiency of the chosen current
state only; it does not prove fundamental non-Markovity.

## 6. Baseline tournament

Finite baselines are ARX, VAR, FIR, finite-dimensional linear state space, and a
Kalman-style latent-state model. Standard non-Markov baselines are fractional
difference/ARFIMA, generalized-Langevin kernel fitting, Volterra series,
predictive-state representation, and reservoir computing.

The primary score is paired held-out interventional NMSE with parameter count, state
dimension, runtime proxy, long-history prediction NMSE, log likelihood, kernel error,
and response-geometry similarity reported separately. A novel residual would require a
relative improvement above the best standard baseline greater than 0.10 with bootstrap
95% lower bound above 0.10. Equal or better standard performance forces
`STANDARD_NONMARKOV_THEORY_SUFFICIENT`.

## 7. Pseudo-memory adversary

Run 500 random and 300 adaptive evolutionary trials across the eight hard-null
families. The detector must not count known finite-order saturation, grid exhaustion,
or detected nonstationarity as target long memory. The preregistered false-positive
rate gate is at most 0.05. Any excess yields `PSEUDOMEMORY_FALSE_POSITIVE` with priority
over positive scientific labels.

## 8. Seeds, freeze, and holdout

- Development: seeds 30000-30029.
- Validation: seeds 31000-31049.
- Confirmation: seeds 40000-40299, opened once.
- Bootstrap RNG: 908172, independent of simulation seeds.

Validation freezes generator grids, history lengths, lag/state grids, model families,
metrics, thresholds, search spaces, holdout definitions, and verdict priority in
`results/timeless_T8_config.json`. Its canonical JSON SHA-256 is recorded before
confirmation. Confirmation refuses to run without a matching freeze and passing
ground-truth/positive-control audit.

After confirmation, open exactly once: unseen power-law alpha, unseen fractional order,
unseen generalized-Langevin kernel, very-high finite AR, large hidden Markov model,
nonlinear Volterra truth, unseen slow-common-driver null, and unseen paired-pulse
separation. No model, metric, threshold, or search space changes after opening.

## 9. Primary gates and verdict priority

1. G1: all finite-Markov positive controls saturate.
2. G2: at least one long-memory target passes the scaling-inefficiency gate.
3. G3: the history-conditioned target passes and the sufficient-state control remains null.
4. G4: pseudo-memory false-positive rate is at most 0.05.
5. G5: all standard non-Markov baselines receive the same data budget.
6. G6: the primary result survives unseen history/generator parameters.
7. G7: no major H1-H8 holdout failure.

Priority is `INCONCLUSIVE_IMPLEMENTATION`, `PSEUDOMEMORY_FALSE_POSITIVE`,
`INCONCLUSIVE_IDENTIFIABILITY`, `FINITE_MARKOV_COMPRESSION_SUFFICIENT`,
`STANDARD_NONMARKOV_THEORY_SUFFICIENT`, `FINITE_MARKOV_COMPRESSION_INEFFICIENT`, then
`NONMARKOV_RESIDUAL_DETECTED`.

`NONMARKOV_RESIDUAL_DETECTED` additionally requires every standard non-Markov baseline
to fail, the best standard baseline to be beaten by the frozen margin, and replication
on fresh seeds and unseen families. Even then the residual is not named Correlon.

