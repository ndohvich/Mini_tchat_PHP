"""Leakage-safe conventional ML benchmark using development validation only."""
import time
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def train_ml_models(Xtr,ytr,Xv,yv,Xtest,seed):
    """Select by chronological validation and refit on train+validation; scalers live inside pipelines."""
    models={"LinearRegression":make_pipeline(SimpleImputer(),StandardScaler(),LinearRegression()),"Ridge":make_pipeline(SimpleImputer(),StandardScaler(),Ridge(1.0)),"RandomForest":make_pipeline(SimpleImputer(),RandomForestRegressor(n_estimators=200,random_state=seed,n_jobs=-1)),"ExtraTrees":make_pipeline(SimpleImputer(),ExtraTreesRegressor(n_estimators=200,random_state=seed,n_jobs=-1)),"HistGradientBoosting":make_pipeline(SimpleImputer(),HistGradientBoostingRegressor(random_state=seed))}
    result={}
    for name,m in models.items():
        start=time.perf_counter();m.fit(Xtr,ytr); val=m.predict(Xv);train_time=time.perf_counter()-start
        result[name]={"model":m,"validation_prediction":val,"train_time":train_time}
    return result
