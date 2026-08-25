"""Diebold-Mariano comparison and moving-block bootstrap metric intervals."""
import numpy as np
from scipy.stats import t
from .metrics import calculate_metrics

def diebold_mariano(y,p1,p2,horizon=1):
    """HAC variance DM test on aligned squared-error differentials."""
    d=(np.asarray(y)-np.asarray(p1))**2-(np.asarray(y)-np.asarray(p2))**2;n=len(d); lag=max(horizon-1,0); var=np.var(d,ddof=1)
    for k in range(1,lag+1): var+=2*(1-k/(lag+1))*np.cov(d[:-k],d[k:],ddof=1)[0,1]
    stat=d.mean()/np.sqrt(var/n) if var>0 else np.nan; return {"DM_statistic":stat,"p_value":float(2*t.sf(abs(stat),n-1)) if np.isfinite(stat) else np.nan}
def moving_block_bootstrap(y,p,iterations,denominator,seed=42):
    """Generate moving-block bootstrap CIs, preserving local temporal dependence."""
    rng=np.random.default_rng(seed);n=len(y);b=max(2,int(np.sqrt(n))); vals=[]
    for _ in range(iterations):
        idx=np.concatenate([np.arange(s,s+b)%n for s in rng.integers(0,n,size=int(np.ceil(n/b)))])[:n]; vals.append(calculate_metrics(np.asarray(y)[idx],np.asarray(p)[idx],denominator))
    return {m:np.quantile([v[m] for v in vals],[.025,.975]).tolist() for m in ["MAE","RMSE","nRMSE"]}
