"""Non-destructive data-quality audit with statistical and limited physical flags."""
import numpy as np,pandas as pd

def audit_data_quality(df, datetime_column, target, config):
    """Flag rather than silently remove outliers; identify temporal gaps from observed timestamps."""
    out=df.copy(); numeric=out.select_dtypes(include="number").columns
    if len(numeric):
        med=out[numeric].median(); mad=(out[numeric]-med).abs().median().replace(0,np.nan)
        robust=((out[numeric]-med).abs().div(1.4826*mad)>3.5).any(axis=1)
        q1,q3=out[numeric].quantile(.25),out[numeric].quantile(.75); iqr=q3-q1
        iqr_flag=((out[numeric]<q1-1.5*iqr)|(out[numeric]>q3+1.5*iqr)).any(axis=1)
        out["is_statistical_outlier"]=(robust|iqr_flag).fillna(False)
    else: out["is_statistical_outlier"]=False
    physical=pd.Series(False,index=out.index); notes=["Unité non démontrée dans le dataset ; contrôle physique complet non appliqué."]
    if target in out: physical|=pd.to_numeric(out[target],errors="coerce")<0; notes.append("Negative target values flagged as physical anomalies.")
    for key in ["wind_speed"]:
        col=config.get("columns",{}).get(key)
        if col in out: physical|=pd.to_numeric(out[col],errors="coerce")<0
    out["is_physical_anomaly"]=physical.fillna(False)
    times=out[datetime_column].dropna().sort_values(); diffs=times.diff().dropna(); freq=diffs.mode().iloc[0] if not diffs.empty else None
    gaps=[]
    if freq is not None:
        for a,b in zip(times.iloc[:-1],times.iloc[1:]):
            if b-a>freq:
                for missing in pd.date_range(a+freq,b-freq,freq=freq): gaps.append({"missing_timestamp":missing,"gap_start":a,"gap_end":b})
    return out,{"duplicate_rows":int(out.duplicated().sum()),"duplicate_timestamps":int(out[datetime_column].duplicated().sum()),"missing":out.isna().sum().to_dict(),"dominant_frequency":str(freq) if freq is not None else None,"statistical_outliers":int(out.is_statistical_outlier.sum()),"physical_anomalies":int(out.is_physical_anomaly.sum()),"notes":notes},pd.DataFrame(gaps)
