# Timeless Correlator T4-T7 - Preregistration

**Version:** 1.0  
**Program:** Order-Duration-Arrow Identifiability Falsification Program  
**Source branch:** `agent/timeless-correlon-phase-t0-t3`  
**Source HEAD:** `a727f11e95e40d7b0919dc87dd9730bfc16abc69`  
**Working branch:** `agent/timeless-correlator-order-duration-arrow-t4-t7`

## 1. Repository continuity audit

The source HEAD contains the frozen T0-T3 preregistration, results, code, raw data,
configuration hash, and independent validator. Those files and verdicts are inherited
without modification:

- T0: `C_ONLY_TIME_NO_GO`.
- T1: `RESTRICTED_GIBBS_GAUSSIAN_RECOVERY`, with canonical `J` supplied externally.
- T2: `MODULAR_FLOW_CONTROL_PASS`, a known-theorem control.
- T3: `CORRELON_EQUALS_CCA_FOR_THIS_OPERATOR`, with no Correlon-specific advantage.

The supplied Phase R report is contextual evidence only. Its frozen conclusion is
`EXISTING_THEORY_SUFFICIENT`: the interventional response is real, while ARX/FIR and
standard causal-dynamical models explain it at least as well. T4-T7 do not reopen that
result.

## 2. Scientific question and claim firewall

The program asks how much temporal structure is identifiable after external clock
coordinates are withheld. It separates:

- order: a precedence or partial-order relation;
- duration: a relative interval, up to one positive scale and an origin when applicable;
- arrow: an orientation distinguishing forward and reversed histories.

Success on one component is not evidence for the other two. The phrases "time emerges
from correlation", "physical time is derived", "causality emerges", and "spacetime
emerges" are prohibited conclusions.

## 3. Information budgets

| Budget | Estimator input | Explicitly withheld | Target |
|---|---|---|---|
| B0 | equal-time stationary correlator hierarchy | timestamps, lags, order, derivatives, current, generator | flow |
| B1 | directed intervention-response relation | timestamps, sample indices, true node order, generator | partial order |
| B2 | randomly serialized transition matrices | durations, timestamps, serialization order labels | relative duration |
| B3 | B2-style symmetric information plus an explicitly supplied current/asymmetry channel | timestamps and hidden generation order | orientation |

Ground truth is isolated from estimator inputs and opened only for scoring.

## 4. Frozen hypotheses and primary gates

### T4 - full equal-time no-go

Use stationary two-dimensional OU pairs
`A0=-gamma I` and `A1=-gamma I+Omega J`, with diffusion `2 gamma I` and stationary
law `N(0,I)`. Population equal-time moments through degree 6 must agree exactly, the
generator norm difference must be positive for `Omega>0`, and the rotational-current
norm must be positive only in the second world. Passing yields
`FULL_EQUAL_TIME_NO_GO`. Finite-sample moment error is diagnostic and cannot overturn
the analytic counterexample.

### T5 - order identifiability

Use linear/nonlinear/branching DAGs, mediated chains, common drivers, disconnected
components, feedback pairs, and directed cycles at sizes 4, 8, 16, and 32 where
defined. Node labels and storage order are independently permuted per seed. For DAGs,
the primary gate is median reachability F1 at least 0.90. For cyclic worlds, cycle
detection accuracy must be at least 0.95 and false-total-order rate at most 0.05.
If the recovered object is ordinary graph reachability/topological structure and does
not beat the same-budget baseline, the verdict is `CAUSAL_ORDER_ONLY`, not temporal
emergence.

### T6 - relative duration identifiability

Use unlabeled operator families `M(tau)=exp(A tau)` in stable, non-normal,
damped-rotation, multi-rate, periodic, nearly-degenerate, and multi-clock worlds. On
identifiable stable worlds, the median positive-affine aligned error must be at most
0.10 and median rank-order accuracy at least 0.90. Exact periodic aliases with unequal
ground-truth durations force `DURATION_ALIASING`. Matrix-log branch ambiguity is
reported separately. A multi-clock one-dimensional embedding residual above 0.20
forces `SCALAR_TIME_MODEL_REJECTED` for that world. Standard matrix-log/semigroup
methods receive exactly B2.

### T7 - arrow identifiability

Use forward Markov chains and stationary time-reversed twins. Symmetric information is
matched by construction. With symmetric input only, the 95% confidence interval of
accuracy must include 0.5 and AUC must be near chance (0.45 to 0.55). With signed
probability current, accuracy and AUC must be at least 0.90. Passing yields
`ARROW_REQUIRES_ASYMMETRY`; a same-budget current or detailed-balance baseline prevents
a Correlon-specific novelty claim.

## 5. Seed protocol and confirmation discipline

- Development: seeds 0-29, implementation and numerical-stability work only.
- Validation: seeds 1000-1049, used to freeze metrics, thresholds, model parameters,
  transformations, holdouts, and verdict priority.
- Confirmation: seeds 20000-20299, opened exactly once after a reproducible SHA-256
  freeze exists.
- Bootstrap RNG: 881122, independent of every simulation seed.

The frozen configuration is `results/timeless_T4T7_config.json`; its canonical JSON
SHA-256 is stored in `results/timeless_T4T7_validation.json`. Confirmation must refuse
to run if either file is absent or if the hash differs. Existing confirmation outputs
are never overwritten.

## 6. Clock-leakage audit

Timestamps and duration labels are absent from estimator records. Row order, node
labels, filenames, serialization order, and opaque operator IDs are randomized. Tests
must demonstrate that predictions are invariant to storage permutation and relabeling.
An adversary using metadata-only features must remain within a binomial 95% chance
interval. Semantic recovery from operator entries is the preregistered T6 target and
is not classified as metadata leakage.

Any unresolved timestamp, duration-label, or storage-order leak forces
`CLOCK_LEAKAGE` and stops confirmation.

## 7. Standard baselines

- order: topological sort, causal reachability, spectral seriation;
- duration: matrix logarithm, semigroup fit, ordinal embedding;
- arrow: detailed-balance test, entropy-production statistic, probability current;
- inherited dynamics context: ARX, VAR, state space, and transfer operators.

Equal or better same-budget standard performance forces `STANDARD_THEORY_SUFFICIENT`
for novelty claims.

## 8. Final untouched holdout

After confirmation is frozen, open exactly once: H1 unseen same-distribution/different
flow, H2 unseen nonlinear DAG, H3 out-of-range periodic alias, H4 unseen multi-clock
rate ratios, H5 directed cycle, H6 unseen time-reversed transition family, and H7 a
long-memory diagnostic. No threshold, model, or metric may change after opening.

## 9. Verdict priority

`CLOCK_LEAKAGE` > `INCONCLUSIVE_IMPLEMENTATION` > `FULL_EQUAL_TIME_NO_GO` >
`SCALAR_TIME_MODEL_REJECTED` > `DURATION_ALIASING` > `GENERATOR_AMBIGUITY` >
`THERMODYNAMIC_ARROW_SUFFICIENT` > `ARROW_REQUIRES_ASYMMETRY` >
`CAUSAL_ORDER_ONLY` > `STANDARD_THEORY_SUFFICIENT` > `RELATIONAL_TIME_CANDIDATE`.

`RELATIONAL_TIME_CANDIDATE` requires all order, duration, orientation, leakage,
generalization, cycle, multi-clock, reverse-twin, and standard-baseline gates. Any one
failure prohibits that label.

## 10. Required reporting

Report per-seed rows, mean, median, standard deviation, 95% confidence interval, fifth
percentile, and failure count. Use paired seeds for estimator/baseline comparisons.
Confirmation and holdout decisions are independently recomputed from raw result files.

