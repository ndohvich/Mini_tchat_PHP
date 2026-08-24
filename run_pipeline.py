"""Single executable entry point for the solar-radiation experimental pipeline."""
import logging, numpy as np, pandas as pd
from src.config import load_config
from src.utils import ensure_directories,setup_logging,set_global_seed,save_environment,write_json
from src.data_loading import load_dataset,detect_column,dataset_inventory
from src.preprocessing import prepare_dataframe,aggregate_if_requested,add_daylight_flag
from src.data_quality import audit_data_quality
from src.statistical_analysis import descriptive_analysis
from src.temporal_analysis import analyze_temporal
from src.feature_engineering import build_features
from src.splitting import chronological_split
from src.baselines import baseline_predictions
from src.ml_models import train_ml_models
from src.catboost_model import train_catboost
from src.statistical_models import train_statistical_models
from src.gfm_models import run_gfm_models
from src.physical_models import physical_clear_sky
from src.evaluation import evaluate_predictions
from src.statistical_tests import moving_block_bootstrap
from src.robustness import robustness_by_month
from src.visualization import create_core_figures
from src.reporting import generate_report

def main():
    ensure_directories();log=setup_logging(); cfg=load_config();set_global_seed(cfg['random']['seed']);save_environment(cfg)
    print('[01/21] Loading dataset...'); raw=load_dataset(cfg['data']['path'],cfg['data'].get('sheet_name'));raw.to_csv('data/interim/raw_snapshot.csv',index=False);write_json('results/reports/dataset_inventory.json',dataset_inventory(raw))
    target,_=detect_column(raw.columns,cfg['data'].get('target_column'),cfg['data']['target_candidates'],'target'); dt,_=detect_column(raw.columns,cfg['data'].get('datetime_column'),cfg['data']['datetime_candidates'],'datetime')
    print('[02/21] Preparing chronological data...'); df,prep=prepare_dataframe(raw,dt,cfg);df,agg=aggregate_if_requested(df,dt,target,cfg)
    print('[03/21] Running data quality audit...');df,quality,gaps=audit_data_quality(df,dt,target,cfg);gaps.to_csv('results/tables/temporal_gaps.csv',index=False);df,daymsg=add_daylight_flag(df,dt,cfg['columns'].get('sunrise'),cfg['columns'].get('sunset'))
    print('[04/21] Descriptive and temporal analysis...');desc,pearson,spearman,mi=descriptive_analysis(df,target);desc.to_csv('results/tables/descriptive_statistics.csv');pearson.to_csv('results/tables/pearson.csv');spearman.to_csv('results/tables/spearman.csv');mi.to_csv('results/tables/mutual_information.csv',index=False);write_json('results/reports/temporal_analysis.json',analyze_temporal(df[target],quality.get('dominant_frequency')))
    print('[05/21] Creating causal features...'); feat=build_features(df,dt,target,cfg);feature_cols=[c for c in feat.select_dtypes(include='number') if c not in [target,'is_statistical_outlier','is_physical_anomaly']]; model=feat.dropna(subset=feature_cols+[target]).reset_index(drop=True)
    splits=chronological_split(model,cfg);write_json('results/reports/splits.json',{k:v.tolist() for k,v in splits.items()});tr,va,te=(splits[k] for k in ['train','validation','test']);X=model[feature_cols];y=model[target].to_numpy(); predictions={};limitations=[daymsg]
    print('[06/21] Training baselines...'); dev=np.r_[tr,va];predictions.update(baseline_predictions(y[dev],y[te]))
    print('[07/21] Training statistical models...'); stats=train_statistical_models(y[tr],y[va]);
    # Statistical candidates are selected on validation; each successful model is then refit solely on development data.
    for name,item in stats.items():
        if 'skipped' in item:
            limitations.append(f'{name} skipped: {item["skipped"]}'); continue
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            m=ARIMA(y[dev],order=(1,0,1)).fit() if name.startswith('ARIMA') else ExponentialSmoothing(y[dev],trend='add',seasonal=None).fit()
            predictions[name]=np.asarray(m.forecast(len(te)))
        except (ValueError,RuntimeError, np.linalg.LinAlgError) as exc: limitations.append(f'{name} final fit skipped: {exc}')
    print('[08/21] Training machine-learning models...'); ml=train_ml_models(X.iloc[tr],y[tr],X.iloc[va],y[va],X.iloc[te],cfg['random']['seed'])
    for name,item in ml.items():
        # Selection used validation only; refit occurs on development period, never locked test.
        item['model'].fit(X.iloc[dev],y[dev]);predictions[name]=item['model'].predict(X.iloc[te])
    cat,msg=train_catboost(X.iloc[tr],y[tr],X.iloc[va],y[va],cfg['random']['seed']);limitations += [msg] if msg else []
    if cat: cat['model'].fit(X.iloc[dev],y[dev],verbose=False);predictions['CatBoost']=cat['model'].predict(X.iloc[te])
    physical,msg=physical_clear_sky(model[dt].iloc[te],cfg);limitations += [msg] if msg else []
    if physical is not None: predictions['PhysicalClearSky']=physical
    _,gfm_notes=run_gfm_models();limitations.extend(gfm_notes.values())
    print('[09/21] Evaluating locked test...'); timestamps=model[dt].iloc[te].to_numpy();daylight=model['is_daylight'].iloc[te].to_numpy() if 'is_daylight' in model else None;performance=evaluate_predictions(predictions,y[te],timestamps,cfg,daylight);performance.to_csv('results/tables/final_locked_test_performance.csv',index=False)
    if predictions:
        best=performance.loc[performance.RMSE.idxmin(),'model']; ci=moving_block_bootstrap(y[te],predictions[best],cfg['evaluation']['bootstrap_iterations'],cfg['evaluation']['nrmse_denominator'],cfg['random']['seed']);write_json('results/reports/bootstrap_ci.json',{best:ci});robustness_by_month(y[te],predictions[best],timestamps,cfg['evaluation']['nrmse_denominator']).to_csv('results/tables/robustness.csv',index=False)
    print('[10/21] Creating figures and report...');create_core_figures(df,dt,target,pearson,performance);df.to_csv('data/processed/model_input.csv',index=False);generate_report({'path':cfg['data']['path'],'rows':len(df),'datetime':dt,'target':target,'quality':quality,'performance':performance,'limitations':limitations});print('Completed. See results/reports/scientific_audit.md')
if __name__=='__main__': main()
