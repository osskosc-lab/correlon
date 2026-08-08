# Phase 2B second-candidate family: frustrated J1-J2 spin chain
# Exact diagonalization in fixed Sz=0, higher-order bond-correlation source,
# projection of conventional Sz_q and bond_q controls, Lanczos spectrum,
# eta scaling, and finite-size diagnostics.

import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
from itertools import combinations
from pathlib import Path
OUT=Path('results'); OUT.mkdir(exist_ok=True)

def basis_bits(L):
    arr=[]
    for occ in combinations(range(L),L//2):
        b=0
        for j in occ: b|=1<<j
        arr.append(b)
    return arr

def build_j1j2(L,J1=1.0,J2=0.5):
    B=basis_bits(L); ix={b:i for i,b in enumerate(B)}; rows=[]; cols=[]; data=[]
    for a,b in enumerate(B):
        diag=0.0
        for r,J in [(1,J1),(2,J2)]:
            for j in range(L):
                k=(j+r)%L; sj=.5 if (b>>j)&1 else -.5; sk=.5 if (b>>k)&1 else -.5; diag+=J*sj*sk
                if ((b>>j)&1)!=((b>>k)&1):
                    rows.append(ix[b^(1<<j)^(1<<k)]); cols.append(a); data.append(.5*J)
        rows.append(a); cols.append(a); data.append(diag)
    return sp.coo_matrix((data,(rows,cols)),shape=(len(B),len(B))).tocsr(),B

def features(B,L):
    sz=np.array([[.5 if (b>>j)&1 else -.5 for j in range(L)] for b in B]); return sz,sz*np.roll(sz,-1,axis=1)
def excitation(vals,g): return vals*g-g*np.dot(g,vals*g)
def project(v,controls):
    nv=np.dot(v,v); Q=[]
    for c in controls:
        z=c.copy()
        for q in Q: z-=q*np.dot(q,z)
        n=np.linalg.norm(z)
        if n>1e-12: Q.append(z/n)
    z=v.copy()
    for _ in range(2):
        for q in Q: z-=q*np.dot(q,z)
    return z,np.dot(z,z)/(nv+1e-30),len(Q)
def source(g,B,L,q=np.pi):
    sz,bond=features(B,L); p=g*g; phase=np.cos(q*np.arange(L))/np.sqrt(L); szq=excitation(sz@phase,g); mean=p@bond; bc=bond-mean[None,:]; bq=excitation(bc@phase,g); corr=np.zeros(len(B))
    for j in range(L): corr+=phase[j]*bc[:,j]*bc[:,(j+2)%L]
    return szq,bq,excitation(corr,g)
def lanczos(H,E0,v,m=140):
    norm=np.linalg.norm(v); q=v/norm; qp=np.zeros_like(q); al=[]; be=[]; bp=0.
    for it in range(min(m,H.shape[0]-1)):
        z=H@q-E0*q; a=np.dot(q,z); z-=a*q
        if it: z-=bp*qp
        z-=q*np.dot(q,z)
        if it: z-=qp*np.dot(qp,z)
        b=np.linalg.norm(z); al.append(a)
        if it<min(m,H.shape[0]-1)-1: be.append(b)
        if b<1e-12: break
        qp,q=q,z/b; bp=b
    T=np.diag(al)
    if len(al)>1:
        bb=np.array(be[:len(al)-1]); T+=np.diag(bb,1)+np.diag(bb,-1)
    e,U=np.linalg.eigh(T); w=norm**2*U[0]**2; keep=(e>1e-8)&(w>1e-14); return e[keep],w[keep]
def eta_fit(e,w):
    f=w/w.sum(); etas=np.linspace(.04,.30,10); grid=np.linspace(max(0,e.min()-.5),min(e.max()+.5,8),5001); H=[]
    for eta in etas: H.append(np.sum(f[:,None]*(eta/np.pi)/((grid[None,:]-e[:,None])**2+eta**2),axis=0).max())
    H=np.array(H); alpha=float(-np.polyfit(np.log(etas),np.log(H),1)[0])
    def atom(x,A,B): return A/x+B
    def cusp(x,A,mu,B): return A*x**(-mu)+B
    models=[]
    for name,fun,p0,bounds,k in [('delta_plus_bg',atom,[.05,.05],([0,0],[np.inf,np.inf]),2),('fractional_cusp',cusp,[.1,.5,.01],([0,0,0],[np.inf,1,np.inf]),3)]:
        p,_=curve_fit(fun,etas,H,p0=p0,bounds=bounds,maxfev=100000); pred=fun(etas,*p); rss=np.sum((H-pred)**2); n=len(H); aic=n*np.log(max(rss/n,1e-300))+2*k; aicc=aic+2*k*(k+1)/(n-k-1); models.append((name,aicc,p))
    d={x[0]:x for x in models}; best=min(models,key=lambda x:x[1]); mu=float(d['fractional_cusp'][2][1]); return alpha,best[0],mu,float(d['delta_plus_bg'][1]-d['fractional_cusp'][1])
def run(L,J2):
    H,B=build_j1j2(L,J2=J2); E,g=eigsh(H,k=1,which='SA',tol=1e-11,maxiter=20000); E0=float(E[0]); g=g[:,0]/np.linalg.norm(g[:,0]); szq,bq,cq=source(g,B,L); v,ret,rank=project(cq,[szq,bq]); e,w=lanczos(H,E0,v); f=w/w.sum(); j=np.argmax(f); topE=e[j]; es=np.sort(e); k=np.argmin(abs(es-topE)); ngh=[]
    if k: ngh.append(topE-es[k-1])
    if k<len(es)-1: ngh.append(es[k+1]-topE)
    alpha,best,mu,daic=eta_fit(e,w); return dict(L=L,J2=J2,dim=H.shape[0],projection_retention=ret,control_rank=rank,top_fraction=float(f.max()),spectral_neff=float(1/np.sum(f*f)),top_energy=float(topE),local_line_gap=float(min(ngh)) if ngh else np.nan,eta_alpha=alpha,best_model=best,mu=mu,delta_minus_cusp_AICc=daic,n_lines=len(e))

rows=[run(L,J2) for J2 in [0.,.3,.5,.7] for L in [10,12,14,16]]; df=pd.DataFrame(rows); df.to_csv(OUT/'Phase2B_J1J2_scan.csv',index=False)
s=[]
for J2,g in df.groupby('J2'):
    s.append(dict(J2=J2,beta_top_fraction=float(-np.polyfit(np.log(g.L),np.log(g.top_fraction),1)[0]),gamma_gap=float(-np.polyfit(np.log(g.L),np.log(g.local_line_gap),1)[0]),mean_mu=float(g.mu.mean()),all_alpha_gate=bool((g.eta_alpha>=.8).all())))
pd.DataFrame(s).to_csv(OUT/'Phase2C_finite_size_summary.csv',index=False)
print(df.to_string(index=False))
