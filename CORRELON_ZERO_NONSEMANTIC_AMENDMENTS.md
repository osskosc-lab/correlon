# Correlon Zero Non-Semantic Implementation Amendments

## Amendment A1 — pandas column accessor after raw confirmation

- Time of discovery: after confirmatory seeds `10000..10199` had all completed and all three raw confirmatory CSV checkpoints had been written.
- Frozen scientific commit: `cb07d84`.
- Frozen scientific implementation hash: `efb4511ff21476614d01ec9449f24b4f3d92deabdd35995ae997cf376503b16c`.
- Failure: `summarize_confirmatory` used `transform.transform.isin(...)`. In pandas, `DataFrame.transform` resolves to the method rather than the column named `transform`, raising `AttributeError` before summary calculation.
- Repair: replace that accessor with `transform["transform"].isin(...)` and add an aggregate-only CLI stage that reads the already-saved raw CSV files.
- Scientific effect: none. No generator, target/null label, seed, score, threshold, transformation membership, transformation output, baseline, retention, raw CSV row, or decision threshold changed. Confirmatory worlds are not regenerated.
- Policy consequence: the pre-confirmatory manifest remains versioned as the scientific freeze. Final summaries also record the postprocessing manifest and this amendment.

This amendment is a reporting-path repair, not Correlon Zero v2 and not a post-hoc metric modification.

## Amendment A2 — explicit score-floor diagnostic and figure QA

- Time of discovery: during independent report and visual inspection after the first complete confirmatory and adversarial summaries.
- Observation: P1 and P4 had formal failure rates of `0.436667`, but direct inspection found a maximum absolute transformed-minus-base score difference of `9.60e-15` across all 600 target-world cases, below the `1e-12` numerical-equivalence tolerance. The 262 failures were numerically zero base-score cases for which the frozen retention formula assigns `0 / max(0,1e-12) = 0`.
- Repair: retain the preregistered decision and precedence unchanged, add machine-readable score-equality diagnostics, state the zero-floor caveat in all decision artifacts, and repair overlapping figure titles/labels.
- Scientific effect: none. No raw result, generator, score, threshold, transformation, baseline, seed, decision condition, failure flag, or primary label changed.
- Interpretation constraint: `FALSIFIED_REPRESENTATION_DEPENDENCE` remains the formal frozen label, but the data do not show a P1/P4 score change beyond floating-point tolerance. The empirical defect is target-score floor/sensitivity plus the separately observed null, baseline, and adversarial failures.

This amendment prevents a misleading interpretation of the frozen rule; it does not rescue or revise v1.
