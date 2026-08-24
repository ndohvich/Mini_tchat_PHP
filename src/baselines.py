"""Causal univariate forecasting baselines aligned to a held-out block."""
import numpy as np

def baseline_predictions(train, test, seasonal_period=None):
    """Generate persistence, historical mean, causal rolling mean, drift, and supported seasonal-naive forecasts."""
    history=list(map(float,train)); out={k:[] for k in ["Naive","HistoricalMean","RollingMean","Drift"]}
    if seasonal_period and len(history)>=seasonal_period: out["SeasonalNaive"]=[]
    for actual in test:
        out["Naive"].append(history[-1]);out["HistoricalMean"].append(float(np.mean(history)));out["RollingMean"].append(float(np.mean(history[-min(7,len(history)):])))
        out["Drift"].append(history[-1] if len(history)<2 else history[0]+(len(history))*(history[-1]-history[0])/(len(history)-1))
        if "SeasonalNaive" in out: out["SeasonalNaive"].append(history[-seasonal_period])
        history.append(float(actual)) # Sequential simulation observes target only after forecast origin.
    return {k:np.array(v) for k,v in out.items()}
