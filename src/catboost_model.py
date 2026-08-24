"""Optional CatBoost validation tuning without access to locked test outcomes."""
import time

def train_catboost(Xtr,ytr,Xv,yv,seed):
    """Train optional CatBoost with temporal validation early stopping; return a documented skip on absence."""
    try: from catboost import CatBoostRegressor
    except ImportError: return None,"CatBoost skipped: optional dependency catboost is not installed."
    start=time.perf_counter(); model=CatBoostRegressor(iterations=1000,depth=6,learning_rate=.05,l2_leaf_reg=3,loss_function="RMSE",random_seed=seed,verbose=False)
    model.fit(Xtr,ytr,eval_set=(Xv,yv),early_stopping_rounds=100,verbose=False)
    return {"model":model,"validation_prediction":model.predict(Xv),"train_time":time.perf_counter()-start,"parameters":model.get_params()},None
