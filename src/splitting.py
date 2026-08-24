"""Strict chronological train/validation/locked-test splitting."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def chronological_split(df,config):
    """Split ordered rows without shuffling; final test block is never tuning input."""
    n=len(df); s=config["split"]; total=s["train_ratio"]+s["validation_ratio"]+s["test_ratio"]
    if abs(total-1)>1e-8 or n<10: raise ValueError("Split ratios must sum to 1 and at least 10 feature rows are required.")
    a=int(n*s["train_ratio"]); b=a+int(n*s["validation_ratio"])
    return {"train":np.arange(a),"validation":np.arange(a,b),"test":np.arange(b,n)}
def temporal_folds(n,config):
    """Return expanding TimeSeriesSplit folds over development data only."""
    return list(TimeSeriesSplit(n_splits=min(config["split"]["n_splits"],max(2,n//3))).split(np.arange(n)))
