#!/usr/bin/env python3
"""
Re-analysis of actual correlon Phase 1I / 1J results CSVs.
Confirms the fractional ridge (mu ≈ 0.5) conclusions from the repository data.
"""

import pandas as pd
from io import StringIO

# Actual data from results/Phase1I_momentum_wide_models.csv and Phase1J_interaction_sweep.csv
phase1i = """U,q_over_pi,best_model,mu,delta_minus_cusp_AICc,smooth_minus_cusp_AICc
0,0.1666666666666666,fractional_cusp,0.328860,14.565253,14.126202
0,0.3333333333333333,fractional_cusp,0.300273,10.158643,1.859041
0,0.5,fractional_cusp,0.293611,10.174497,2.272841
0,0.6666666666666666,smooth_continuum,0.316540,8.453068,-2.041235
0,0.8333333333333334,smooth_continuum,0.346805,8.071576,-1.482454
0,1.0,fractional_cusp,0.360134,8.974025,2.415183
8,0.1666666666666666,fractional_cusp,0.487143,10.524643,18.509173
8,0.3333333333333333,fractional_cusp,0.529577,16.167139,29.765620
8,0.5,fractional_cusp,0.494727,12.279366,19.890026
8,0.6666666666666666,fractional_cusp,0.505883,12.523294,21.253682
8,0.8333333333333334,fractional_cusp,0.480468,10.076503,14.451202
8,1.0,fractional_cusp,0.482291,10.176983,15.039645
"""

phase1j = """L,U,E0,projection_retention,top_fraction,spectral_neff,best_model,mu,delta_minus_cusp_AICc,smooth_minus_cusp_AICc
10,0.0,-12.784906,0.828683,0.088365,20.967499,fractional_cusp,0.357826,11.771019,9.626029
10,2.0,-8.517316,0.785854,0.136552,15.369725,fractional_cusp,0.427753,9.146478,7.482439
10,4.0,-5.802351,0.888598,0.151969,12.742191,fractional_cusp,0.469540,8.207482,7.830522
10,8.0,-3.314309,0.972597,0.186287,8.489524,fractional_cusp,0.458631,6.734454,2.478790
"""

def main():
    df_i = pd.read_csv(StringIO(phase1i))
    df_j = pd.read_csv(StringIO(phase1j))

    print("=== Re-analysis of actual Phase 1I (momentum-wide) ===")
    u8 = df_i[df_i.U == 8]
    u0 = df_i[df_i.U == 0]
    print(f"U/t=8: all best_model == fractional_cusp? {(u8.best_model == 'fractional_cusp').all()}")
    print(f"U/t=8: mu mean ± std = {u8.mu.mean():.3f} ± {u8.mu.std(ddof=1):.3f}")
    print(f"U/t=8: mu range = {u8.mu.min():.3f} – {u8.mu.max():.3f}")
    print(f"U/t=8: all ΔAICc(delta-cusp) > 10? {(u8.delta_minus_cusp_AICc > 10).all()}")
    print(f"U/t=0: mu mean = {u0.mu.mean():.3f}")
    print()

    print("=== Re-analysis of actual Phase 1J (interaction sweep) ===")
    print(df_j[["U", "mu", "best_model", "delta_minus_cusp_AICc", "spectral_neff"]].to_string(index=False))
    print()
    print(f"mu increases with U but stays << 1: {df_j.mu.min():.3f} → {df_j.mu.max():.3f}")
    print(f"All cases prefer fractional_cusp? {(df_j.best_model == 'fractional_cusp').all()}")
    print()

    print("=== Conclusion from actual repository results ===")
    print("Phase 1I & 1J data confirm:")
    print("- Momentum-wide fractional cusp with mu ≈ 0.50 ± 0.02 at U/t=8")
    print("- Interaction sharpens the response (mu rises) but does not produce a delta pole")
    print("- Consistent with an interaction-enhanced fractional singular continuum ridge,")
    print("  not an isolated Correlon quasiparticle pole.")

if __name__ == "__main__":
    main()
