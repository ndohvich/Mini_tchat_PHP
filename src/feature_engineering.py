"""Causal calendar, lag, and rolling feature construction."""
import numpy as np,pandas as pd

def build_features(df, datetime_column,target,config):
    """Create features available at forecast origin; target history is always shifted first."""
    out=df.copy(); dt=out[datetime_column]; f=config["features"]
    if f.get("use_temporal_features"):
        out["hour"],out["day_of_week"],out["day_of_year"],out["month"],out["year"]=dt.dt.hour,dt.dt.dayofweek,dt.dt.dayofyear,dt.dt.month,dt.dt.year
    if f.get("use_cyclical_features"):
        for c,period in [("hour",24),("day_of_year",365.25),("month",12)]:
            if c in out: out[c+"_sin"]=np.sin(2*np.pi*out[c]/period);out[c+"_cos"]=np.cos(2*np.pi*out[c]/period)
    if f.get("use_lag_features"):
        for lag in range(1,int(f.get("max_lag",3))+1): out[f"{target}_lag_{lag}"]=out[target].shift(lag)
    if f.get("use_rolling_features"):
        prior=out[target].shift(1) # Prevents current/future target leakage.
        for w in f.get("rolling_windows",[3,7]):
            out[f"{target}_rolling_mean_{w}"]=prior.rolling(w,min_periods=w).mean();out[f"{target}_rolling_std_{w}"]=prior.rolling(w,min_periods=w).std()
    c=config.get("columns",{});
    if f.get("interactions") and c.get("temperature") in out and c.get("humidity") in out: out["temperature_x_humidity"]=out[c["temperature"]]*out[c["humidity"]]
    return out
