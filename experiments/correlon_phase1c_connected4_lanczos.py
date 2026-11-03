import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from itertools import combinations
from pathlib import Path
from scipy.stats import t as student_t

OUT=Path('/mnt/data/correlon_phase1c')
OUT.mkdir(exist_ok=True)

def basis_bits(L,n):
    out=[]
    for occ in combinations(range(L),n):
        b=0
        for j in occ: b |= 1<<j
        out.append(b)
    return out

def hop_sign(bits,i,j):
    if i==j: return 1
    lo,hi=sorted((i,j)); mask=0
    for k in range(lo+1,hi): mask |= 1<<k
    return -1 if ((bits & mask).bit_count() & 1) else 1

def spin_hop_twist(L,n,t=1.0,phi=np.pi/2):
    B=basis_bits(L,n); ix={b:i for i,b in enumerate(B)}
    rows=[]; cols=[]; data=[]
    peierls=np.exp(1j*phi/L)
    for a,bits in enumerate(B):
        for j in range(L):
            k=(j+1)%L
            if ((bits>>j)&1) and not ((bits>>k)&1):
                nb=bits^(1<<j)^(1<<k)
                rows.append(ix[nb]); cols.append(a)
                data.append(-t*peierls*hop_sign(bits,k,j))
            if ((bits>>k)&1) and not ((bits>>j)&1):
                nb=bits^(1<<k)^(1<<j)
                rows.append(ix[nb]); cols.append(a)
                data.append(-t*np.conj(peierls)*hop_sign(bits,j,k))
    return sp.coo_matrix((data,(rows,cols)),shape=(len(B),len(B)),dtype=np.complex128).tocsr(),B

def build_hubbard(L,U,t=1.0,phi=np.pi/2):
    n=L//2
    K,B=spin_hop_twist(L,n,t,phi)
    nb=len(B); I=sp.eye(nb,format='csr',dtype=np.complex128)
    H=sp.kron(K,I,format='csr')+sp.kron(I,K,format='csr')
    diag=np.empty(nb*nb,float); p=0
    for ub in B:
        for db in B:
            diag[p]=U*((ub&db).bit_count()); p+=1
    H=H+sp.diags(diag,format='csr')
    return H,B

def sector_bilinear(L,n,r,q):
    B=basis_bits(L,n); ix={b:i for i,b in enumerate(B)}
    rows=[]; cols=[]; data=[]
    for a,bits in enumerate(B):
        for j in range(L):
            k=(j+r)%L
            phase=np.exp(-1j*q*(j+0.5*r))/np.sqrt(L)
            if r==0:
                if (bits>>j)&1:
                    rows.append(a); cols.append(a); data.append(phase)
            else:
                if ((bits>>j)&1) and not ((bits>>k)&1):
                    nb=bits^(1<<j)^(1<<k)
                    rows.append(ix[nb]); cols.append(a)
                    data.append(phase*hop_sign(bits,k,j))
    return sp.coo_matrix((data,(rows,cols)),shape=(len(B),len(B)),dtype=np.complex128).tocsr()

def apply_spin_even_onebody(M,g,nb):
    G=g.reshape(nb,nb)
    Y=M@G + G@M.T
    y=np.asarray(Y).reshape(-1)
    y=y-g*np.vdot(g,y)
    return y

def occupation_matrix(B,L):
    occ=np.zeros((len(B),L),dtype=np.float64)
    for i,b in enumerate(B):
        for j in range(L): occ[i,j]=(b>>j)&1
    return occ

def local_expectations_fast(g,occ):
    nb=len(occ); prob=np.abs(g.reshape(nb,nb))**2
    pu=prob.sum(axis=1); pdn=prob.sum(axis=0)
    return pu@occ,pdn@occ

def composite_diag_values_fast(g,B,L,q,kind,occ=None,means=None):
    nb=len(B)
    if occ is None: occ=occupation_matrix(B,L)
    if means is None: means=local_expectations_fast(g,occ)
    upm,dnm=means
    A=occ-upm[None,:]; D=occ-dnm[None,:]
    phase=np.exp(-1j*q*np.arange(L))/np.sqrt(L)
    if kind=='doublon_c':
        V=(A*phase[None,:])@D.T
    elif kind in ('nn_charge_c','nn_spin_c'):
        row=np.zeros(nb,dtype=np.complex128); col=np.zeros(nb,dtype=np.complex128)
        cross=np.zeros((nb,nb),dtype=np.complex128)
        sign=1.0 if kind=='nn_charge_c' else -1.0
        for j in range(L):
            jp=(j+1)%L; ph=phase[j]
            row += ph*A[:,j]*A[:,jp]
            col += ph*D[:,j]*D[:,jp]
            cross += sign*ph*(np.outer(A[:,j],D[:,jp])+np.outer(A[:,jp],D[:,j]))
        V=row[:,None]+col[None,:]+cross
    else: raise ValueError(kind)
    return V.reshape(-1)

def corr_of_corr_diag_values(g,B,L,q,occ=None,means=None):
    nb=len(B)
    if occ is None: occ=occupation_matrix(B,L)
    if means is None: means=local_expectations_fast(g,occ)
    upm,dnm=means
    A=occ-upm[None,:]; D=occ-dnm[None,:]
    prob=np.abs(g.reshape(nb,nb))**2
    C=[]; cm=[]
    for j in range(L):
        jp=(j+1)%L
        cj=(A[:,j]*A[:,jp])[:,None] + (D[:,j]*D[:,jp])[None,:] + np.outer(A[:,j],D[:,jp]) + np.outer(A[:,jp],D[:,j])
        C.append(cj.astype(np.float32)); cm.append(float(np.sum(prob*cj)))
    phase=np.exp(-1j*q*np.arange(L))/np.sqrt(L)
    V=np.zeros((nb,nb),dtype=np.complex128)
    for j in range(L):
        k=(j+2)%L
        V += phase[j]*(C[j].astype(np.float64)-cm[j])*(C[k].astype(np.float64)-cm[k])
    return V.reshape(-1)

def orthonormalize_controls(vectors):
    Q=[]
    for c in vectors:
        c=c.astype(np.complex128,copy=True)
        for q in Q: c-=q*np.vdot(q,c)
        nc=np.linalg.norm(c)
        if nc>1e-10: Q.append(c/nc)
    return Q

def excitation_from_diag(vals,g):
    v=vals*g
    return v-g*np.vdot(g,v)

def build_onebody_Q(g,B,L,q):
    nb=len(B); n=L//2; Q=[]
    for r in range(L):
        M=sector_bilinear(L,n,r,q)
        c=apply_spin_even_onebody(M,g,nb)
        for qq in Q: c -= qq*np.vdot(qq,c)
        nc=np.linalg.norm(c)
        if nc>1e-10:
            c=(c/nc).astype(np.complex64)
            for qq in Q: c -= qq*np.vdot(qq,c)
            nc2=np.linalg.norm(c)
            if nc2>1e-8: Q.append((c/nc2).astype(np.complex64))
    return Q

def project_with_Q(v,Q):
    nv=float(np.vdot(v,v).real); vr=v.astype(np.complex128,copy=True)
    for _ in range(2):
        for qq in Q:
            q=qq.astype(np.complex128,copy=False); vr -= q*np.vdot(q,vr)
    nr=float(np.vdot(vr,vr).real)
    return vr,nr/(nv+1e-30),len(Q)

def lanczos_measure(H,E0,v,m=80,tol=1e-12):
    norm=np.linalg.norm(v)
    if norm<1e-14: return pd.DataFrame(columns=['energy','weight','fraction'])
    q=v/norm; qprev=np.zeros_like(q); alpha=[]; beta=[]; bprev=0.0
    for it in range(m):
        z=H@q - E0*q
        a=np.vdot(q,z).real
        z=z-a*q-bprev*qprev
        z=z-q*np.vdot(q,z)
        b=np.linalg.norm(z); alpha.append(float(a))
        if it<m-1: beta.append(float(b))
        if b<tol: break
        qprev,q=q,z/b; bprev=b
    n=len(alpha); T=np.diag(alpha)
    if n>1:
        bb=np.asarray(beta[:n-1]); T+=np.diag(bb,1)+np.diag(bb,-1)
    e,U=np.linalg.eigh(T); w=(norm**2)*np.abs(U[0,:])**2
    keep=(e>1e-9)&(w>1e-14); e=e[keep]; w=w[keep]
    if len(e)==0: return pd.DataFrame(columns=['energy','weight','fraction'])
    order=np.argsort(e); e=e[order]; w=w[order]
    groups=[]
    for ee,ww in zip(e,w):
        if groups and abs(ee-groups[-1][0])<1e-7:
            e0,w0=groups[-1]; groups[-1]=((e0*w0+ee*ww)/(w0+ww),w0+ww)
        else: groups.append((ee,ww))
    df=pd.DataFrame(groups,columns=['energy','weight']); df['fraction']=df.weight/df.weight.sum()
    return df.sort_values('weight',ascending=False).reset_index(drop=True)

def spectral_metrics(lines):
    if len(lines)==0: return dict(top_energy=np.nan,top_fraction=0.0,spectral_neff=np.nan,total_weight=0.0,n_lines=0)
    p=lines.fraction.to_numpy()
    return dict(top_energy=float(lines.iloc[0].energy),top_fraction=float(lines.iloc[0].fraction),spectral_neff=float(1/np.sum(p*p)),total_weight=float(lines.weight.sum()),n_lines=int(len(lines)))

def run_case(L,U,phi=np.pi/2,q=np.pi):
    H,B=build_hubbard(L,U,phi=phi)
    E,g=eigsh(H,k=1,which='SA',tol=1e-9,maxiter=5000)
    E0=float(E[0].real); g=g[:,0].astype(np.complex128); g/=np.linalg.norm(g)
    m=80 if L<=10 else 52
    occ=occupation_matrix(B,L); means=local_expectations_fast(g,occ); Q1=build_onebody_Q(g,B,L,q)
    kinds=['doublon_c','nn_charge_c','nn_spin_c']; vres={}; rets={}; ranks={}
    for kind in kinds:
        vals=composite_diag_values_fast(g,B,L,q,kind,occ=occ,means=means)
        v=excitation_from_diag(vals,g); vr,ret,rank=project_with_Q(v,Q1)
        vres[kind]=vr; rets[kind]=ret; ranks[kind]=rank
    Qknown=orthonormalize_controls([vres[k] for k in kinds])
    qvals=corr_of_corr_diag_values(g,B,L,q,occ=occ,means=means); qv=excitation_from_diag(qvals,g)
    qvr,qret1,_=project_with_Q(qv,Q1); qnorm1=float(np.vdot(qvr,qvr).real); qvr2=qvr.copy()
    for qc in Qknown: qvr2-=qc*np.vdot(qc,qvr2)
    qret_known=float(np.vdot(qvr2,qvr2).real/(qnorm1+1e-30))
    vres['corr2_c']=qvr2; rets['corr2_c']=qret1*qret_known; ranks['corr2_c']=len(Q1)
    rows=[]
    for kind in kinds+['corr2_c']:
        lines=lanczos_measure(H,E0,vres[kind],m=m); met=spectral_metrics(lines)
        row={'L':L,'U':U,'phi_over_pi':phi/np.pi,'q_over_pi':q/np.pi,'operator':kind,'dim':H.shape[0],'E0':E0,'onebody_rank':ranks[kind],'known_quartic_rank':len(Qknown) if kind=='corr2_c' else 0,'residual_retention':rets[kind],'post_known_retention':qret_known if kind=='corr2_c' else np.nan}
        row.update(met); rows.append(row)
    return rows
