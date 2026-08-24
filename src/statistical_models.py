"""Guarded univariate ARIMA and exponential-smoothing models."""
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def train_statistical_models(train,validation):
    """Fit compact compatible models, retaining errors rather than aborting the pipeline."""
    out={}
    for name,fit in {"ARIMA(1,0,1)":lambda:ARIMA(train,order=(1,0,1)).fit(),"ETS":lambda:ExponentialSmoothing(train,trend="add",seasonal=None).fit()}.items():
        try:
            m=fit();out[name]={"model":m,"validation_prediction":np.asarray(m.forecast(len(validation))),"parameters":{"aic":getattr(m,"aic",None)}}
        except (ValueError,RuntimeError, np.linalg.LinAlgError) as e: out[name]={"skipped":str(e)}
    return out
