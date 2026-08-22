# Correlon Zero — Anti-Correlon Adversarial Audit

## Frozen objective

Maximize the frozen `CorrelonZero_v1` score subject to `direct_XY_edge=false`. The search used seed `424242`, 300 random-search trials, and no metric or threshold update.

## Result

- Frozen threshold: `0.0511`
- Maximum null score: `0.7431`
- Crossing count: `82 / 300`
- Repeated-crossing gate: `FAIL` for v1
- Best parameters: `{'drift': 0.2629403015043294, 'frequency': 0.2804527699694389, 'latent_rank': 1, 'noise': 0.1422200660077279, 'rho': 0.6954582476999487, 'sample_length': 600, 'strength': 0.553927167239739, 'x_delay': 2, 'y_delay': 2}`

| trial | score | crosses | rank | rho | strength | noise | T |
|---|---|---|---|---|---|---|---|
| 266 | 0.7431 | True | 1 | 0.695 | 0.554 | 0.142 | 600 |
| 215 | 0.7225 | True | 1 | 0.898 | 2.321 | 0.531 | 600 |
| 134 | 0.7172 | True | 1 | 0.296 | 2.479 | 0.367 | 600 |
| 92 | 0.6878 | True | 1 | 0.913 | 2.558 | 0.993 | 600 |
| 161 | 0.6649 | True | 1 | 0.044 | 2.545 | 1.054 | 600 |
| 6 | 0.6322 | True | 1 | 0.583 | 2.655 | 0.328 | 750 |
| 59 | 0.6161 | True | 1 | 0.199 | 2.969 | 0.614 | 300 |
| 139 | 0.5683 | True | 1 | 0.196 | 2.604 | 0.175 | 300 |
| 39 | 0.5629 | True | 1 | 0.517 | 2.338 | 0.628 | 750 |
| 43 | 0.5589 | True | 1 | 0.106 | 1.840 | 1.058 | 450 |

## Mechanism firewall

Candidates were generated only from exogenous latent factors, optional oscillatory persistence, independent observation noise, deterministic drift, and observation delays. Neither X nor Y was used to update the other. The best-case archive is `results/correlon_zero_best_adversarial.npz`.

## Interpretation

An adversarial crossing is a false positive for the frozen operational criterion, not evidence that the null secretly contains the target edge. The result is permanent for v1 and cannot be repaired on these seeds.
