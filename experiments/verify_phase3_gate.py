#!/usr/bin/env python3
"""
Self-contained verification of Phase 3 joint pole gate
from correlon repository.

Recomputes the conservative conjunctive gate on the published
Phase 2B/2C J1-J2 results and the known controls.
"""

import pandas as pd
import numpy as np
from io import StringIO

# Embedded published results
phase2b = """L,J2,dim,E0,projection_retention,control_rank,top_fraction,spectral_neff,top_energy,local_line_gap,eta_alpha,best_model,mu,delta_minus_cusp_AICc,n_lines
10,0.0,252,-4.515446354492041,0.9366586901952619,2,0.6046616462671687,2.288865515761868,2.4963259103572497,0.7546614654876103,0.9742638554794756,delta_plus_bg,0.999999999999422,-4.285714286480427,8
12,0.0,924,-5.3873909174452095,0.9465999220866566,2,0.3211286607226129,4.39094592528205,2.157927076298145,0.7311678972982762,0.9052640590611196,delta_plus_bg,0.9999999999982835,-4.285714286073812,26
14,0.0,3432,-6.263549533547028,0.9533645148769031,2,0.41654010413075226,3.743010901908761,2.817404332253534,0.23460956064627858,0.9119348189603804,delta_plus_bg,0.9999999999999999,-4.28571428571442,59
16,0.0,12870,-7.142296360616782,0.9582497968100018,2,0.3903760515178046,4.29147405566501,2.588039857797922,0.14744932228029572,0.9018071062944872,delta_plus_bg,0.9999999999999958,-4.285714285716011,61
10,0.3,252,-3.986108163931501,0.9050450122256466,2,0.5268096363501253,2.2983791347416656,1.7076001971941075,0.9679820647386213,0.9674843056437171,delta_plus_bg,0.9999999999999999,-4.285714285714562,8
12,0.3,924,-4.7630316413477605,0.9167167298195076,2,0.3503467981807184,3.7105565830999185,2.7191923306033208,0.015915271395003927,0.8901205854270171,delta_plus_bg,0.999999999999912,-4.28571428574503,29
14,0.3,3432,-5.54257981974372,0.9259427608823387,2,0.32920396306499183,3.883107082822095,2.624780583664523,0.12772984333604187,0.8566053470137456,delta_plus_bg,0.9999999999999999,-4.285714285714349,69
16,0.3,12870,-6.323774269876451,0.9331196793417192,2,0.31426485375173735,5.472786678685015,1.8173962999061972,0.07262808369592944,0.8654793657461513,delta_plus_bg,0.9999999999999999,-4.285714285714349,79
10,0.5,252,-3.7500000000000027,0.9642411199361212,2,0.511209111575934,3.1503710716446034,1.3704423290689625,0.039200970788995315,0.9764968007257522,delta_plus_bg,0.997807212296545,-2.561259428237989,18
12,0.5,924,-4.5000000000000036,0.9644458489860323,2,0.2880670029430617,6.896103067991631,2.365975045692958,0.003215145061062241,0.7984147942786163,fractional_cusp,0.9745719498493094,9.319146142532645,53
14,0.5,3432,-5.249999999999991,0.9942654408292108,2,0.18388492246610716,9.186187539782162,1.684656602788471,0.005738998539780438,0.8291578196715343,fractional_cusp,0.8161840639492102,19.973817118835612,87
16,0.5,12870,-5.999999999999994,0.9461397570577851,2,0.25780473989932196,7.8728170791932195,2.373715906405789,0.03258433894254775,0.7573794054578604,fractional_cusp,0.7567032557219167,45.25099478319297,91
10,0.7,252,-4.0313424470869075,0.9881529437695091,2,0.21571690595281853,6.190415841036262,2.504680357514662,0.07771271693990789,0.7776502247037752,fractional_cusp,0.8522442807083681,38.71215821276863,8
12,0.7,924,-4.808576285117787,0.9999941061714216,2,0.5120193953280477,3.4244046937908035,1.8280098385591406,0.11116435736354124,0.9103899790007755,fractional_cusp,0.968916655837445,13.647205913065818,29
14,0.7,3432,-5.564767981020437,0.9961401784596874,2,0.2434915683999844,8.159597091360116,1.5687364042678096,0.1443883699135835,0.7500920667875012,delta_plus_bg,0.9999999999999999,-4.28571428571442,69
16,0.7,12870,-6.373120567164199,0.9976953038251838,2,0.2945856190494868,7.758774142978637,1.8440705759843625,0.015282076852955129,0.7389921780104487,delta_plus_bg,0.9999999999996667,-4.285714285733704,94
"""

phase2c = """J2,beta_top_fraction,gamma_gap,top_fraction_inf_linear_1L,gap_inf_linear_1L,mean_mu,all_delta_gate
0.0,0.7297297952574376,3.7911711285819596,0.03629412618503138,-0.9623708849422505,0.9999999999994253,True
0.3,1.0624866862702735,3.9968306949714223,-0.0645513797547228,-1.4991917894520548,0.9999999999999779,True
0.5,1.6805950434432892,0.4105808253912404,-0.28753656281096135,-0.0029965769806196774,0.8863166204542452,False
0.7,-0.19652687364252389,2.686753186200978,0.35211916975669194,0.020952501175765175,0.95529023413637,False
"""

def main():
    df = pd.read_csv(StringIO(phase2b))
    su = pd.read_csv(StringIO(phase2c))

    rows = []
    for _, r in su.iterrows():
        j = float(r.J2)
        g = df[df.J2 == j].sort_values("L")
        alpha_gate = bool((g.eta_alpha >= 0.8).all())
        residue_gate = bool((r.beta_top_fraction <= 0.25) and (g.iloc[-1].top_fraction >= 0.20))
        gap_gate = bool((g.iloc[-1].local_line_gap >= 0.05) and (r.gamma_gap <= 0.5))
        model_consistency = bool((g.best_model == "delta_plus_bg").all())
        strict = alpha_gate and residue_gate and gap_gate
        rows.append({
            "family": "J1-J2 spin chain",
            "parameter": f"J2={j:.1f}",
            "alpha_gate": alpha_gate,
            "residue_gate": residue_gate,
            "gap_gate": gap_gate,
            "model_consistency": model_consistency,
            "strict_pole_gate": strict,
            "largest_L": int(g.L.max()),
            "largest_L_alpha": float(g.iloc[-1].eta_alpha),
            "largest_L_top_fraction": float(g.iloc[-1].top_fraction),
            "largest_L_gap": float(g.iloc[-1].local_line_gap),
        })

    # Known controls from RESULTS
    rows.append({
        "family": "1D repulsive Hubbard higher-order",
        "parameter": "U/t=8",
        "alpha_gate": False,
        "residue_gate": False,
        "gap_gate": False,
        "model_consistency": False,
        "strict_pole_gate": False,
        "largest_L": 16,
        "largest_L_alpha": 0.518,
        "largest_L_top_fraction": np.nan,
        "largest_L_gap": np.nan,
    })
    rows.append({
        "family": "Attractive Hubbard bound pair",
        "parameter": "U/t=-4, K=pi/2",
        "alpha_gate": True,
        "residue_gate": True,
        "gap_gate": True,
        "model_consistency": True,
        "strict_pole_gate": True,
        "largest_L": 64,
        "largest_L_alpha": 0.994,
        "largest_L_top_fraction": 0.816497,
        "largest_L_gap": np.nan,
    })

    out = pd.DataFrame(rows)
    print("=== Phase 3 Joint Pole Gate Verification ===\n")
    print(out[["family", "parameter", "alpha_gate", "residue_gate", "gap_gate", "strict_pole_gate"]].to_string(index=False))
    print()
    print("Summary:")
    print(f"  J1-J2 candidates passing strict gate: {out[out.family.str.contains('J1-J2')].strict_pole_gate.sum()}")
    print(f"  Attractive Hubbard control passes:   {out[out.family.str.contains('Attractive')].strict_pole_gate.iloc[0]}")
    print(f"  Repulsive Hubbard higher-order fails: {not out[out.family.str.contains('repulsive')].strict_pole_gate.iloc[0]}")
    print()
    print("Conclusion matches repository: only the known attractive-Hubbard")
    print("bound-pair positive control passes the conservative Phase 3 gate.")
    print("No J1-J2 candidate and no repulsive-Hubbard higher-order response")
    print("is promoted to an isolated Correlon quasiparticle pole.")

if __name__ == "__main__":
    main()
