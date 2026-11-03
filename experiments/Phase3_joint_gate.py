import pandas as pd, numpy as np
from pathlib import Path
OUT=Path('results')
df=pd.read_csv(OUT/'Phase2B_J1J2_scan.csv'); su=pd.read_csv(OUT/'Phase2C_finite_size_summary.csv')
rows=[]
for _,r in su.iterrows():
    j=float(r.J2); g=df[df.J2==j].sort_values('L')
    alpha_gate=bool((g.eta_alpha>=0.8).all())
    residue_gate=bool((r.beta_top_fraction<=0.25) and (g.iloc[-1].top_fraction>=0.20))
    gap_gate=bool((g.iloc[-1].local_line_gap>=0.05) and (r.gamma_gap<=0.5))
    model_consistency=bool((g.best_model=='delta_plus_bg').all())
    rows.append(dict(family='J1-J2 spin chain',parameter=f'J2={j:.1f}',alpha_gate=alpha_gate,residue_gate=residue_gate,gap_gate=gap_gate,model_consistency=model_consistency,strict_pole_gate=alpha_gate and residue_gate and gap_gate,largest_L=int(g.L.max()),largest_L_alpha=float(g.iloc[-1].eta_alpha),largest_L_top_fraction=float(g.iloc[-1].top_fraction),largest_L_gap=float(g.iloc[-1].local_line_gap),beta_top_fraction=float(r.beta_top_fraction),gamma_gap=float(r.gamma_gap)))
rows += [dict(family='1D repulsive Hubbard higher-order',parameter='U/t=8',alpha_gate=False,residue_gate=False,gap_gate=False,model_consistency=False,strict_pole_gate=False,largest_L=16,largest_L_alpha=.518,largest_L_top_fraction=np.nan,largest_L_gap=np.nan,beta_top_fraction=np.nan,gamma_gap=np.nan),dict(family='Attractive Hubbard bound pair',parameter='U/t=-4,K=pi/2',alpha_gate=True,residue_gate=True,gap_gate=True,model_consistency=True,strict_pole_gate=True,largest_L=64,largest_L_alpha=.994,largest_L_top_fraction=.816497,largest_L_gap=np.nan,beta_top_fraction=0.,gamma_gap=0.)]
out=pd.DataFrame(rows); out.to_csv(OUT/'Phase3_joint_decision_matrix.csv',index=False); print(out.to_string(index=False))
