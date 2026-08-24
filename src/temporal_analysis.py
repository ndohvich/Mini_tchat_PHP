"""Data-driven time-series diagnostics with guarded statistical tests."""
import pandas as pd
from statsmodels.tsa.stattools import adfuller,kpss

def analyze_temporal(series,frequency=None):
    """Run stationarity tests only where sample size supports them."""
    y=pd.Series(series).dropna(); result={"n":len(y),"frequency":str(frequency)}
    if len(y)>=20:
        try: result["adf_pvalue"]=float(adfuller(y,autolag="AIC")[1])
        except ValueError as e: result["adf_error"]=str(e)
        try: result["kpss_pvalue"]=float(kpss(y,nlags="auto")[1])
        except ValueError as e: result["kpss_error"]=str(e)
    return result
