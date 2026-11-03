# Reproducibility notes

## Repository-local reruns

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The interaction sweep is repository-local:

```bash
python experiments/Phase1J_interaction_sweep_portable.py
```

The physical Phase 2 bridge positive control is also self-contained:

```bash
python experiments/Phase2_bridge_positive_control.py
```

## Phase 1G-1I provenance

Phase 1G-1I are post-processing falsification tests applied to the converged Krylov tridiagonal coefficients and L=12 momentum-resolved line spectra produced in Phases 1D/1F. The analysis implementation is in `experiments/Phase1G_1I_analysis.py`; final outputs are summarized in `RESULTS.md` and `results/`.

The full user-delivered reproducibility package also contains the exact Phase 1F Krylov input and Phase 1D momentum-line inputs used for the reported Phase 1G-1I run. These large intermediate experiment products are intentionally kept outside the minimal GitHub branch rather than duplicated into source control.

## Interpretation discipline

- A surviving projected higher-order sector is not counted as a Correlon quasiparticle.
- A finite-size Lanczos line spacing is not counted as a thermodynamic continuum gap.
- Pole confirmation requires resolution-stable Ritz convergence, delta-like broadening scaling, nonzero thermodynamic residue, and separation from the continuum threshold.
- Synthetic controls and the attractive-Hubbard physical bound-pair control are mandatory calibration checks.
