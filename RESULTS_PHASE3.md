# Correlon continuous falsification: Phase 2 -> Phase 3

## Verdict

Phase 3 is reached. The strong **Correlon quasiparticle / isolated delta-pole** claim is **not promoted to conditional support**.

The Phase 1 repulsive-Hubbard candidate already failed the pole gate. In the independent Phase 2B J1-J2 spin-chain family, no tested coupling (`J2/J1 = 0, 0.3, 0.5, 0.7`, `L=10-16`) passes the conservative joint pole gate. The known attractive-Hubbard bound-pair positive control remains the only system that passes.

## Phase 2A - calibration falsification

A synthetic calibration suite tested smooth continua, square-root cusps, delta+continuum mixtures, finite comb spectra, and near-degenerate doublets.

The simple rule `alpha >= 0.8 AND delta_plus_bg best` had sensitivity `0.000` and specificity `0.875` on this stress test. Delta+continuum spectra were absorbed by the fractional-cusp model with `mu -> 1`, while sparse finite-comb spectra could generate false pole calls.

Conclusion: AICc delta-vs-cusp model selection is useful diagnostic evidence but is not a universal pole classifier. Phase 3 therefore uses a conjunctive gate requiring broadening scaling, finite-size residue stability, and an isolated continuum gap.

## Phase 2B - second candidate family

A periodic spin-1/2 J1-J2 Heisenberg chain was studied in the fixed `Sz=0` sector. The higher-order source was a bond correlation-of-correlations operator after projection of conventional `Sz_q` and `bond_q` controls.

Projection retention remained high (`~0.905-1.000`), confirming a nontrivial higher-order response sector. However, a residual higher-order sector is not sufficient evidence of a quasiparticle.

At the largest size `L=16`:

| J2/J1 | alpha | top fraction | local line gap | strict pole gate |
|---:|---:|---:|---:|---|
| 0.0 | 0.902 | 0.390 | 0.1474 | FAIL |
| 0.3 | 0.865 | 0.314 | 0.0726 | FAIL |
| 0.5 | 0.757 | 0.258 | 0.0326 | FAIL |
| 0.7 | 0.739 | 0.295 | 0.0153 | FAIL |

The `J2=0` and `0.3` cases are particularly instructive: they look pole-like under eta scaling, but their leading spectral weight and local gap do not remain stable as system size grows. This reproduces the finite-resolution false-pole failure mode found earlier in Phase 1.

## Phase 2C - finite-size falsification

Power-law diagnostics for the leading spectral fraction give approximate exponents `beta = 0.73, 1.06, 1.68` for `J2=0,0.3,0.5`. The `J2=0.7` series is non-monotone and is treated as ambiguous rather than supportive. Local line gaps also shrink strongly in most series.

No stable nonzero thermodynamic residue plus finite continuum gap is demonstrated.

## Phase 3 - joint decision

The conservative Phase 3 gate requires all of:

1. pole-like broadening scaling (`alpha >= 0.8`) across sizes;
2. no clear finite-size collapse of the leading residue, with largest-size fraction >= 0.20;
3. a finite local isolation gap that does not show strong collapse.

All J1-J2 candidates fail at least one required gate. The repulsive-Hubbard higher-order candidate fails. The attractive-Hubbard bound pair passes as the physical positive control.

## Scientific position after Phase 3

Supported weak claim: higher-order response structure can survive projection of conventional lower-order channels.

Unsupported strong claim: such residual structure generically condenses into an isolated Correlon quasiparticle pole.

The quasiparticle part of Correlon v4.0 should therefore be strongly contracted. A data-consistent continuation would treat Correlon as either a higher-order singular-continuum mode or as a persistent inter-space coupling mode rather than assuming a particle interpretation.

## Important limitations

- J1-J2 calculations are finite-size (`L <= 16`) exact-diagonalization/Lanczos studies, not a thermodynamic-limit proof.
- Lanczos was not fully reorthogonalized; local line gaps are diagnostics, not final spectral proof.
- Phase 2A is a fixed-broadening stress test, not a universal benchmark for every spectral estimator.
- A future quasiparticle re-test should use larger systems with MPS correction-vector or Chebyshev spectral methods.
