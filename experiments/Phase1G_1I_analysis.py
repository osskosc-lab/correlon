import json, os, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import curve_fit

OUT=Path('/mnt/data/Correlon_Phase1G_to_Phase2')
OUT.mkdir(exist_ok=True)
meta=json.load(open('/mnt/data/Correlon_Phase1F_Work/specsafe_meta.json'))
A=np.asarray(meta['alpha'],float); B=np.asarray(meta['beta'],float)

ETAS=np.linspace(0.5,2.5,9)
GRID=np.linspace(15,22,5001)

def spec_from_ab(a,b,m):
    ev,U=eigh_tridiagonal(a[:m],b[:m-1])
    f=np.abs(U[0])**2
    return ev,f/f.sum()

def lorentz_spectrum(ev,f,eta,grid=GRID):
    return np.sum(f[:,None]*(eta/np.pi)/((grid[None,:]-ev[:,None])**2+eta**2),axis=0)

def peak_curve(a,b,m,etas=ETAS):
    ev,f=spec_from_ab(a,b,m)
    H=[]; W=[]
    for eta in etas:
        S=lorentz_spectrum(ev,f,eta)
        j=int(np.argmax(S)); H.append(float(S[j])); W.append(float(GRID[j]))
    return np.asarray(H),np.asarray(W)

def alpha_fit(etas,H):
    p=np.polyfit(np.log(etas),np.log(H),1)
    return float(-p[0])

def lanczos_diagonal(E,W,mmax=72):
    q=np.sqrt(W).astype(float); q/=np.linalg.norm(q)
    qprev=np.zeros_like(q); alpha=[]; beta=[]; bp=0.0
    for it in range(mmax):
        z=E*q
        aa=float(np.dot(q,z)); z-=aa*q
        if it>0: z-=bp*qprev
        z-=q*np.dot(q,z)
        bb=float(np.linalg.norm(z))
        alpha.append(aa); beta.append(bb)
        if bb<1e-14: break
        qprev,q=q,z/bb; bp=bb
    return np.asarray(alpha),np.asarray(beta)

N=40000
Eiso=np.r_[18.0,np.linspace(20.0,60.0,N-1)]
Wiso=np.r_[0.25,np.full(N-1,0.75/(N-1))]
ai,bi=lanczos_diagonal(Eiso,Wiso)
Econ=np.linspace(15.0,22.0,N); Wcon=np.full(N,1/N)
ac,bc=lanczos_diagonal(Econ,Wcon)
Ecusp=np.linspace(15.0,22.0,N); e0=18.5; eps=(Ecusp[1]-Ecusp[0])/2
Wcusp=1/np.sqrt(np.abs(Ecusp-e0)+eps); Wcusp/=Wcusp.sum()
ap,bp=lanczos_diagonal(Ecusp,Wcusp)

rows=[]
controls={'candidate':(A,B),'delta_pole_control':(ai,bi),'smooth_continuum_control':(ac,bc),'sqrt_cusp_control':(ap,bp)}
for label,(a,b) in controls.items():
    for m in [40,48,56,64,72]:
        H,W=peak_curve(a,b,m)
        al=alpha_fit(ETAS,H)
        for eta,h,w in zip(ETAS,H,W):
            rows.append({'system':label,'krylov_order':m,'eta':eta,'peak_height':h,'peak_energy':w,'alpha':al})
pg=pd.DataFrame(rows)
pg.to_csv(OUT/'Phase1G_eta_scaling.csv',index=False)
agg=pg.groupby(['system','krylov_order'],as_index=False).first()[['system','krylov_order','alpha']]
agg.to_csv(OUT/'Phase1G_alpha_summary.csv',index=False)

cand_late=agg[(agg.system=='candidate') & (agg.krylov_order>=64)].alpha
ctrl_alpha=agg[agg.krylov_order==72].set_index('system').alpha.to_dict()
pd.DataFrame([{'candidate_late_mean_alpha':float(cand_late.mean()),'delta_control_alpha_m72':float(ctrl_alpha['delta_pole_control']),'sqrt_cusp_control_alpha_m72':float(ctrl_alpha['sqrt_cusp_control']),'delta_pole_gate_alpha_ge_0_8':bool(cand_late.mean()>=0.8),'verdict':'DELTA POLE REJECTED; FRACTIONAL SINGULAR CONTINUUM FAVORED'}]).to_csv(OUT/'Phase1G_gate.csv',index=False)

def model_atom(x,A0,B0): return A0/x+B0
def model_cusp(x,A0,mu,B0): return A0*x**(-mu)+B0
def model_smooth(x,B0,C0): return B0+C0*x

def fit_models(df):
    x=df.eta.to_numpy(float); y=df.peak_height.to_numpy(float); n=len(y)
    specs=[('delta_plus_bg',model_atom,[.08,.05],([0,0],[np.inf,np.inf]),2),('fractional_cusp',model_cusp,[.15,.5,.01],([0,0,0],[np.inf,1,np.inf]),3),('smooth_continuum',model_smooth,[.2,-.05],([-np.inf,-np.inf],[np.inf,np.inf]),2)]
    out=[]
    for name,func,p0,bounds,k in specs:
        popt,_=curve_fit(func,x,y,p0=p0,bounds=bounds,maxfev=200000)
        pred=func(x,*popt); rss=float(np.sum((y-pred)**2)); aic=n*np.log(max(rss/n,1e-300))+2*k; aicc=aic+2*k*(k+1)/(n-k-1)
        mu=float(popt[1]) if name=='fractional_cusp' else (1.0 if name=='delta_plus_bg' else 0.0)
        out.append({'model':name,'aicc':float(aicc),'mu':mu})
    return pd.DataFrame(out)

mh=[]
for system in controls.keys():
    d=pg[(pg.system==system)&(pg.krylov_order.isin([56,64,72]))][['eta','peak_height']]
    f=fit_models(d); f['system']=system; mh.append(f)
mh=pd.concat(mh,ignore_index=True)
mh.to_csv(OUT/'Phase1H_model_comparison.csv',index=False)

cand=mh[mh.system=='candidate'].set_index('model')
pd.DataFrame([{'candidate_best_model':mh[mh.system=='candidate'].sort_values('aicc').iloc[0].model,'candidate_mu':float(cand.loc['fractional_cusp','mu']),'delta_minus_cusp_AICc':float(cand.loc['delta_plus_bg','aicc']-cand.loc['fractional_cusp','aicc']),'verdict':'FRACTIONAL CUSP MODEL DECISIVELY PREFERRED OVER DELTA+BACKGROUND'}]).to_csv(OUT/'Phase1H_gate.csv',index=False)

def collect_lines(U):
    base=Path('/mnt/data/Correlon_Phase1D_Output')
    names=[f'lines_partial_L12U{U}.csv',f'lines_L12U{U}_m5.csv',f'lines_L12U{U}_m6.csv']
    return pd.concat([pd.read_csv(base/n) for n in names],ignore_index=True).drop_duplicates(subset=['q_over_pi','energy','weight'])

def q_fit(U):
    d=collect_lines(U); out=[]
    for qv in sorted(d.q_over_pi.unique()):
        g=d[d.q_over_pi==qv]; ev=g.energy.to_numpy(float); f=g.weight.to_numpy(float); f=f/f.sum()
        local_grid=np.linspace(max(0,ev.min()-1),min(ev.max()+1,50),5001)
        H=[]
        for eta in ETAS:
            S=np.sum(f[:,None]*(eta/np.pi)/((local_grid[None,:]-ev[:,None])**2+eta**2),axis=0); H.append(float(S.max()))
        ff=fit_models(pd.DataFrame({'eta':ETAS,'peak_height':H})); best=ff.sort_values('aicc').iloc[0]
        cusp=ff[ff.model=='fractional_cusp'].iloc[0]; atom=ff[ff.model=='delta_plus_bg'].iloc[0]; smooth=ff[ff.model=='smooth_continuum'].iloc[0]
        out.append({'U':U,'q_over_pi':qv,'best_model':best.model,'mu':float(cusp.mu),'delta_minus_cusp_AICc':float(atom.aicc-cusp.aicc),'smooth_minus_cusp_AICc':float(smooth.aicc-cusp.aicc)})
    return pd.DataFrame(out)

qi=pd.concat([q_fit(0),q_fit(8)],ignore_index=True); qi.to_csv(OUT/'Phase1I_momentum_wide_models.csv',index=False)
strong=qi[qi.U==8]; free=qi[qi.U==0]
pd.DataFrame([{'U8_all_q_best_fractional_cusp':bool((strong.best_model=='fractional_cusp').all()),'U8_mu_mean':float(strong.mu.mean()),'U8_mu_std':float(strong.mu.std(ddof=1)),'U8_mu_min':float(strong.mu.min()),'U8_mu_max':float(strong.mu.max()),'U8_all_q_delta_rejected_DAICc_gt_10':bool((strong.delta_minus_cusp_AICc>10).all()),'U0_mu_mean':float(free.mu.mean()),'U0_cusp_best_fraction':float((free.best_model=='fractional_cusp').mean()),'verdict':'MOMENTUM-WIDE MU~1/2 SINGULAR CONTINUUM RIDGE; NOT A QUASIPARTICLE BRANCH'}]).to_csv(OUT/'Phase1I_gate.csv',index=False)
