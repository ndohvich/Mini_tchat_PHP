"""Transparent regression metrics, including zero-aware MAPE."""
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def calculate_metrics(y_true,y_pred,nrmse_denominator="range"):
    """Calculate aligned finite-observation metrics; nRMSE denominator is explicitly returned."""
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float); mask=np.isfinite(y)&np.isfinite(p); y,p=y[mask],p[mask]
    if not len(y): return {"n_observations":0}
    rmse=float(np.sqrt(mean_squared_error(y,p))); denom=float(np.ptp(y) if nrmse_denominator=="range" else np.mean(np.abs(y)))
    nonzero=np.abs(y)>np.finfo(float).eps
    return {"n_observations":len(y),"MAE":float(mean_absolute_error(y,p)),"MSE":float(mean_squared_error(y,p)),"RMSE":rmse,"R2":float(r2_score(y,p)) if len(y)>1 else np.nan,"nRMSE":rmse/denom if denom else np.nan,"nRMSE_denominator":nrmse_denominator,"sMAPE":float(np.mean(2*np.abs(p-y)/(np.abs(y)+np.abs(p)+np.finfo(float).eps))*100),"MAPE":float(np.mean(np.abs((p[nonzero]-y[nonzero])/y[nonzero]))*100) if nonzero.any() else np.nan,"mape_excluded_zero_count":int((~nonzero).sum())}
