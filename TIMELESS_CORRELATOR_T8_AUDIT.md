# Timeless Correlator T8 - Repository and Inheritance Audit

**Audit version:** 1.0  
**Repository:** `osskosc-lab/correlon`  
**Required source commit:** `f1e0227d44ef50e489c6a32cfc0941dbf5f6925e`  
**Working branch:** `agent/timeless-correlator-t8-nonmarkov-memory`

## Source verification

The source commit exists, is the tip of the completed T4-T7 branch at audit time, and
contains `TIMELESS_CORRELATOR_T4T7_RESULTS.md`, all T4-T7 raw result files, the frozen
configuration, independent confirmation validation, figures, and H1-H7 holdout.

The following inherited verdicts are immutable:

| Phase | Frozen verdict |
|---|---|
| T4 | `FULL_EQUAL_TIME_NO_GO` |
| T5 | `CAUSAL_ORDER_ONLY` |
| T6 primary | `DURATION_ALIASING` |
| T6 secondary | `GENERATOR_AMBIGUITY`, `SCALAR_TIME_MODEL_REJECTED` |
| T7 | `ARROW_REQUIRES_ASYMMETRY` |
| T4-T7 overall | `STANDARD_THEORY_SUFFICIENT` |
| Relational-time candidate | `false` |

T8 is not a rescue experiment. A T8 positive result cannot change any row in this
table, cannot establish temporal emergence, and cannot be named Correlon evidence.

## Reusable infrastructure

T8 reuses only methodological infrastructure from T4-T7:

- deterministic stage-specific seeds;
- development, validation, SHA-256 freeze, one-time confirmation, then holdout;
- raw per-seed CSV results and independent gate recomputation;
- explicit same-information-budget baseline comparisons;
- refusal to overwrite confirmation or holdout outputs;
- claim firewalls and negative-control-first verdict priority;
- noninteractive report and figure generation.

No T4-T7 raw result is used as T8 training data, and no inherited result file is edited.

## T8 scope

T8 tests whether finite Markov compression is efficient over a preregistered resolution
and complexity budget, and whether standard non-Markov models explain any remaining
long-history prediction and intervention response. It does not test whether time or a
new physical primitive emerges.

