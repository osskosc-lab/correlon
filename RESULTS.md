# Correlon continuous falsification: Phase 1G -> Phase 2 bridge

## Final Phase 1 result

The higher-order correlation sector survives projection of the complete spin-even one-body tangent space and known quartic channels, but the surviving response in the 1D half-filled repulsive Hubbard benchmark is **not supported as an isolated Correlon quasiparticle pole**.

The best effective description after Phase 1G-1J is an **interaction-enhanced fractional singular continuum ridge**, with an exponent close to `mu = 1/2`.

## Phase 1G - broadening scaling

- Candidate late-order exponent: `alpha = 0.518`
- Delta-pole control: `alpha = 0.913`
- Smooth-continuum control: `alpha = 0.255`
- Square-root cusp control: `alpha = 0.588`
- Delta-pole gate `alpha >= 0.8`: **FAIL**

## Phase 1H - direct model comparison

- Best model: `fractional_cusp`
- Fitted exponent: `mu = 0.526`
- `Delta AICc(delta - cusp) = 26.2`
- All synthetic calibration controls were classified correctly.

Result: the fractional-cusp model is decisively preferred over delta+background.

## Phase 1I - momentum-wide falsification

For `U/t = 8`, `L = 12`, all six momenta from `q/pi = 1/6` through `1` prefer the fractional-cusp model.

- Mean `mu = 0.497`
- Standard deviation `0.019`
- Range `0.480-0.530`
- Delta model rejected by `Delta AICc > 10` at every momentum.

This is inconsistent with a single quasiparticle dispersion and consistent with a momentum-wide singular continuum ridge.

## Phase 1J - interaction sweep

At `L=10`, `q=pi`:

| U/t | mu | projected retention | spectral N_eff |
|---:|---:|---:|---:|
| 0 | 0.358 | 0.829 | 20.97 |
| 2 | 0.428 | 0.786 | 15.37 |
| 4 | 0.470 | 0.889 | 12.74 |
| 8 | 0.459 | 0.973 | 8.49 |

Interaction sharpens the higher-order singular response but does not drive the exponent toward the delta-pole value `mu=1`.

## Phase 2 connection gate - physical positive control

A two-particle attractive-Hubbard bound pair at `L=64`, `U/t=-4`, `K=pi/2` was used as an interacting physical positive control.

- Analytic bound energy: `-4.898979485566356 t`
- Ritz energy error: `2.7e-15 t`
- Bound spectral fraction: `0.816497`
- Ritz residual: `5.94e-16 t`
- Broadening exponent: `alpha = 0.994`
- Delta model preferred over fractional cusp.

The same pipeline therefore **does recover a real interacting composite bound-state pole**.

## Scientific conclusion

The restricted claim

> The 1D half-filled repulsive Hubbard `U/t=8` higher-order response is an isolated Correlon quasiparticle pole.

is rejected by the Phase 1G-1J falsification sequence.

Correlon Theory v4.0 as a general framework is not globally falsified by this one benchmark family. The Phase 2 entry point is now reached:

1. expand the physical positive-control suite;
2. move to a second candidate family (frustrated spin / spin-liquid or glassy `chi4` higher-order channel);
3. retain the same pole gates: conventional-channel independence, `mu -> 1`, resolution-stable Ritz pole, nonzero thermodynamic residue, and a true continuum threshold gap.
