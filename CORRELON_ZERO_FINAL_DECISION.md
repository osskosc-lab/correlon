# Correlon Zero — Final Decision

## Decision

**FALSIFIED_REPRESENTATION_DEPENDENCE**

This is the formal label required by the frozen precedence. It must not be read as observed basis dependence: P1/P4 maximum absolute score change was `9.600e-15`, below `1e-12`. The formal flag comes entirely from `262` / `600` numerically unchanged `0→0` target cases receiving retention zero.

| criterion | observed | pass |
|---|---|---|
| Delta mean >= 0.10 | -0.9942 | False |
| Delta q05 > 0 | -1.0000 | False |
| Every negative FPR <= 0.05 | 1.000 | False |
| Baseline advantage >= 0.10 | -0.9969 | False |
| P1/P4 frozen-ratio failure rate <= 0.05 | 0.437 | False |
| P1/P4 maximum absolute score change <= 1e-12 | 9.600e-15 | True |
| Adversarial crossings < 3 | 82 | False |

## Failure matrix

- Primary label: `FALSIFIED_REPRESENTATION_DEPENDENCE`
- All failure labels: `['FALSIFIED_REPRESENTATION_DEPENDENCE', 'FALSIFIED_COMMON_DRIVER', 'FALSIFIED_MATCHED_NULL', 'FALSIFIED_BY_BASELINE', 'FALSIFIED_ADVERSARIAL']`
- Strongest baseline: `pca_rank1_explained_variance`
- Strongest named null by FPR: `common_driver` (`1.000`)
- Maximum adversarial null score: `0.7431`
- Empirically strongest independent failures: common-driver FPR `1.000`; matched-low-rank FPR `1.000`; paired baseline advantage `-0.9969`; adversarial crossings `82 / 300`

## Claim boundary

The decision falsifies or supports only `CorrelonZero_v1` as preregistered in commit `8544101` and scientifically frozen in commit `cb07d84`. It does not establish an ontological Correlon, direct causation, or a physical discovery.

## Exact next experiment

Preregister Correlon Zero v2 with two separate gates before using any preserve/destroy ratio: (1) a target-sensitivity gate that rejects a candidate when its untransformed target score is at the floor, and (2) an explicit zero-to-zero equivalence rule for invariance. Then use fresh target seeds and hold out the present common-driver, matched-low-rank, and adversarial-null parameter regions until final evaluation. The present v1 remains rejected and these seeds may not be used to tune v2.
