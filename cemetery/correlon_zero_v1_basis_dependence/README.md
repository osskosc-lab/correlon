# Correlon Zero v1 Cemetery Entry

- metric definition: `sqrt(clip(T_iso,0,1) * clip(T_floor,0,1))`
- preregistration commit: `8544101`
- scientific implementation commit: `cb07d84`
- pre-confirmatory scientific implementation hash: `efb4511ff21476614d01ec9449f24b4f3d92deabdd35995ae997cf376503b16c`
- postprocessing implementation hash: `3449b49161ccf867b39540cf9f6146dfe46f64c59ce238f582427b250a4872cf`
- falsifying experiment: `CORRELON_ZERO_ADVERSARIAL_INVARIANCE_FALSIFICATION_v1.0`
- seed policy: pilot `0..19`; confirmatory `10000..10199`; adversary `424242`
- failure criterion: `FALSIFIED_REPRESENTATION_DEPENDENCE` under the frozen precedence
- all failure labels: `['FALSIFIED_REPRESENTATION_DEPENDENCE', 'FALSIFIED_COMMON_DRIVER', 'FALSIFIED_MATCHED_NULL', 'FALSIFIED_BY_BASELINE', 'FALSIFIED_ADVERSARIAL']`
- result summary: mean Delta `-0.9942`; common-driver FPR `1.000`; matched-low-rank FPR `1.000`; paired baseline advantage `-0.9969`; adversarial crossings `82`
- formal-label caveat: P1/P4 absolute scores matched within `1e-12` (maximum difference `9.600e-15`); `262` / `600` zero-score targets failed only because the frozen ratio maps `0→0` to `0`
- reason for rejection: target-score floor/sensitivity failure, common-driver and matched-null false positives, baseline dominance, and repeated adversarial false positives independently violate necessary v1 criteria; no v1 repair was attempted

See `CORRELON_ZERO_RESULTS.md`, `CORRELON_ZERO_ADVERSARIAL_AUDIT.md`, and `CORRELON_ZERO_FINAL_DECISION.md`.
