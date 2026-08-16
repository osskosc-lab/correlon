import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from pathlib import Path
OUT=Path('results'); OUT.mkdir(exist_ok=True)
etas=np.linspace(.04,.30,10); grid=np.linspace(0,6,2001)

def fit(ev,w):
    w=w/w.sum(); H=[]
    for eta in etas:
        S=np.sum(w[:,None]*(eta/np.pi)/((grid[None,:]-ev[:,None])**2+eta**2),axis=0); H.append(S.max())
    H=np.array(H); alpha=float(-np.polyfit(np.log(etas),np.log(H),1)[0])
    def atom(x,A,B): return A/x+B
    def cusp(x,A,mu,B): return A*x**(-mu)+B
    def smooth(x,B,C): return B+C*x
    out=[]
    for name,fun,p0,bounds,k in [('delta_plus_bg',atom,[.05,.05],([0,0],[np.inf,np.inf]),2),('fractional_cusp',cusp,[.1,.5,.01],([0,0,0],[np.inf,1,np.inf]),3),('smooth_continuum',smooth,[.2,-.05],([-np.inf,-np.inf],[np.inf,np.inf]),2)]:
        p,_=curve_fit(fun,etas,H,p0=p0,bounds=bounds,maxfev=100000); pred=fun(etas,*p); rss=np.sum((H-pred)**2); n=len(H); aic=n*np.log(max(rss/n,1e-300))+2*k; aicc=aic+2*k*(k+1)/(n-k-1); mu=p[1] if name=='fractional_cusp' else (1 if name=='delta_plus_bg' else 0); out.append((name,aicc,mu))
    best=min(out,key=lambda z:z[1]); d={n:(a,m) for n,a,m in out}
    return alpha,best[0],float(d['fractional_cusp'][1]),float(d['delta_plus_bg'][0]-d['fractional_cusp'][0])

def continuum(N=1200,shape='smooth'):
    e=np.linspace(1,5,N)
    if shape=='smooth': w=np.ones(N)
    elif shape=='sqrt_cusp': w=1/np.sqrt(np.abs(e-2.5)+(e[1]-e[0])/2)
    return e,w/w.sum()

def delta_cont(Z):
    e,w=continuum(); return np.r_[2.0,e],np.r_[Z,(1-Z)*w]

def comb(n,seed):
    rr=np.random.default_rng(seed); e=np.sort(rr.uniform(1,5,n)); w=rr.dirichlet(np.ones(n)); return e,w

cases=[]
for rep in range(20):
    for label,truth,gen in [('smooth_continuum','nonpole',lambda:continuum(shape='smooth')),('sqrt_cusp','nonpole',lambda:continuum(shape='sqrt_cusp')),('delta_Z10','pole',lambda:delta_cont(.10)),('delta_Z25','pole',lambda:delta_cont(.25)),('delta_Z50','pole',lambda:delta_cont(.50)),('finite_comb_8','nonpole',lambda rep=rep:comb(8,1000+rep)),('finite_comb_32','nonpole',lambda rep=rep:comb(32,2000+rep)),('doublet','nonunique',lambda rep=rep:(np.array([2.0,2.0+0.03+0.002*rep]),np.array([.5,.5])) )]:
        ev,w=gen(); a,b,m,da=fit(ev,w); pred='pole' if (a>=.8 and b=='delta_plus_bg') else 'nonpole'; cases.append(dict(case=label,truth=truth,rep=rep,alpha=a,best_model=b,mu=m,delta_minus_cusp_AICc=da,pred=pred))
df=pd.DataFrame(cases); df.to_csv(OUT/'Phase2A_calibration_suite.csv',index=False)
summary=df.groupby(['case','truth']).agg(alpha_mean=('alpha','mean'),alpha_sd=('alpha','std'),pole_call_rate=('pred',lambda s:(s=='pole').mean()),delta_best_rate=('best_model',lambda s:(s=='delta_plus_bg').mean()),mu_mean=('mu','mean')).reset_index(); summary.to_csv(OUT/'Phase2A_calibration_summary.csv',index=False)
z=df[df.truth.isin(['pole','nonpole'])]; tp=((z.truth=='pole')&(z.pred=='pole')).sum(); fn=((z.truth=='pole')&(z.pred!='pole')).sum(); fp=((z.truth=='nonpole')&(z.pred=='pole')).sum(); tn=((z.truth=='nonpole')&(z.pred!='pole')).sum(); pd.DataFrame([dict(tp=tp,fn=fn,fp=fp,tn=tn,sensitivity=tp/(tp+fn),specificity=tn/(tn+fp))]).to_csv(OUT/'Phase2A_confusion.csv',index=False)
print(summary.to_string(index=False))
