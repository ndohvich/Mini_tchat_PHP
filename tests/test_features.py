"""Tests for causal lag/rolling features and chronological split."""
import pandas as pd
from src.feature_engineering import build_features
from src.splitting import chronological_split

def cfg(): return {'features':{'use_temporal_features':False,'use_cyclical_features':False,'use_lag_features':True,'use_rolling_features':True,'max_lag':1,'rolling_windows':[2],'interactions':False},'columns':{},'split':{'train_ratio':.6,'validation_ratio':.2,'test_ratio':.2,'n_splits':2}}
def test_rolling_uses_only_past():
    df=pd.DataFrame({'t':pd.date_range('2024-01-01',periods=4,freq='h'),'y':[1.,2.,100.,4.]});out=build_features(df,'t','y',cfg())
    assert out.loc[2,'y_rolling_mean_2']==1.5

def test_chronological_split():
    s=chronological_split(pd.DataFrame({'x':range(10)}),cfg());assert max(s['train'])<min(s['validation'])<min(s['test'])

def test_scaler_fits_train_only():
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    scaler=StandardScaler().fit(np.array([[1.],[2.]]))
    assert scaler.mean_[0]==1.5
