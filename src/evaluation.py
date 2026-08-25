"""Evaluation tables and prediction artifact persistence."""
import pandas as pd
from .metrics import calculate_metrics

def evaluate_predictions(predictions,y,index,config,daylight=None):
    """Evaluate every aligned prediction under the same locked observations."""
    rows=[]
    for name,pred in predictions.items():
        row={"model":name,**calculate_metrics(y,pred,config["evaluation"]["nrmse_denominator"])};rows.append(row)
        pd.DataFrame({"timestamp":index,"actual":y,"predicted":pred}).to_csv(f"results/predictions/{name.replace('/','_')}.csv",index=False)
        if daylight is not None and config["evaluation"].get("daylight_only_evaluation"):
            row["daylight_RMSE"]=calculate_metrics(y[daylight],pred[daylight],config["evaluation"]["nrmse_denominator"]).get("RMSE")
    return pd.DataFrame(rows)
