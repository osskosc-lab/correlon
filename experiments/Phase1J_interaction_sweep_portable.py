import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments'))
import correlon_phase1c_connected4_lanczos as c
from scipy.optimize import curve_fit

OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
ETAS=np.linspace(.5,2.5,9)

def broaden(ev,f,eta,grid): return np.sum(f[:,None]*(eta/np.pi)/((grid[None,:]-ev[:,None])**2+eta**2),axis=0)
def atom(x,A,B): return A/x+B
def cusp(x,A,mu,B): return A*x**(-mu)+B
def smooth(x,B,C): return B+C*x

def fit_models(eta,H):
    n=len(H); out=[]
    for name,fun,p0,bounds,k in [('delta_plus_bg',atom,[.08,.05],([0,0],[np.inf,np.inf]),2),('fractional_cusp',cusp,[.15,.5,.01],([0,0,0],[np.inf,1,np.inf]),3),('smooth_continuum',smooth,[.2,-.05],([-np.inf,-np.inf],[np.inf,np.inf]),2)]:
        p,_=curve_fit(fun,eta,H,p0=p0,bounds=bounds,maxfev=100000)
        pred=fun(eta,*p); rss=np.sum((H-pred)**2); aic=n*np.log(max(rss/n,1e-300))+2*k; aicc=aic+2*k*(k+1)/(n-k-1)
        out.append((name,float(aicc),p))
    return out

def run(U,L=10,q=np.pi):
    H,B=c.build_hubbard(L,U,phi=np.pi/2)
    from scipy.sparse.linalg import eigsh
    E,g=eigsh(H,k=1,which='SA',tol=1e-9,maxiter=5000)
    E0=float(E[0].real); g=g[:,0].astype(np.complex128); g/=np.linalg.norm(g)
    occ=c.occupation_matrix(B,L); means=c.local_expectations_fast(g,occ); Q1=c.build_onebody_Q(g,B,L,q)
    kinds=['doublon_c','nn_charge_c','nn_spin_c']; vres=[]
    for kind in kinds:
        vals=c.composite_diag_values_fast(g,B,L,q,kind,occ=occ,means=means); v=c.excitation_from_diag(vals,g); vr,_,_=c.project_with_Q(v,Q1); vres.append(vr)
    Qknown=c.orthonormalize_controls(vres)
    qvals=c.corr_of_corr_diag_values(g,B,L,q,occ=occ,means=means); qv=c.excitation_from_diag(qvals,g); qvr,qret,_=c.project_with_Q(qv,Q1); qnorm=np.vdot(qvr,qvr).real
    for qc in Qknown: qvr-=qc*np.vdot(qc,qvr)
    knownret=np.vdot(qvr,qvr).real/(qnorm+1e-30)
    lines=c.lanczos_measure(H,E0,qvr,m=80)
    ev=lines.energy.to_numpy(float); f=lines.weight.to_numpy(float); f/=f.sum(); grid=np.linspace(max(0,ev.min()-1),min(ev.max()+1,50),5001)
    heights=np.array([broaden(ev,f,e,grid).max() for e in ETAS])
    fits=fit_models(ETAS,heights); dd={n:(a,p) for n,a,p in fits}; best=min(dd,key=lambda x:dd[x][0]); mu=float(dd['fractional_cusp'][1][1])
    return {'L':L,'U':U,'E0':E0,'projection_retention':float(qret*knownret),'top_fraction':float(lines.fraction.max()),'spectral_neff':float(1/np.sum(lines.fraction.to_numpy()**2)),'best_model':best,'mu':mu,'delta_minus_cusp_AICc':float(dd['delta_plus_bg'][0]-dd['fractional_cusp'][0]),'smooth_minus_cusp_AICc':float(dd['smooth_continuum'][0]-dd['fractional_cusp'][0])}

if __name__=='__main__':
    rows=[run(U) for U in [0.,2.,4.,8.]]
    df=pd.DataFrame(rows); df.to_csv(OUT/'Phase1J_interaction_sweep_reproduced.csv',index=False)
    print(df.to_string(index=False))
