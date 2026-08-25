"""Chronological validation, optional aggregation, and daylight classification."""
import numpy as np, pandas as pd

def prepare_dataframe(df, datetime_column, config):
    """Convert and sort timestamps; invalid timestamps are reported then excluded because chronology is impossible."""
    out=df.copy(); parsed=pd.to_datetime(out[datetime_column],errors="coerce")
    invalid=int(parsed.isna().sum()); out[datetime_column]=parsed
    out=out.dropna(subset=[datetime_column]).sort_values(datetime_column).reset_index(drop=True)
    return out,{"invalid_datetime_rows_excluded":invalid,"sorted_chronologically":True}

def circular_mean(series):
    """Compute wind direction mean on a circle rather than an invalid arithmetic mean."""
    radians=np.deg2rad(pd.to_numeric(series,errors="coerce")); return np.rad2deg(np.arctan2(np.nanmean(np.sin(radians)),np.nanmean(np.cos(radians))))%360

def aggregate_if_requested(df, datetime_column, target, config):
    """Aggregate only when explicitly configured; direction uses circular aggregation."""
    freq=config["time"].get("aggregation_frequency")
    if not freq: return df,{"aggregation":"not requested"}
    indexed=df.set_index(datetime_column); direction=config.get("columns",{}).get("wind_direction")
    numeric=indexed.select_dtypes(include="number").columns
    agg={c:(circular_mean if c==direction else "mean") for c in numeric}
    out=indexed.resample(freq).agg(agg).reset_index()
    return out,{"aggregation_frequency":freq,"rule":"numeric mean; configured wind direction circular mean"}

def add_daylight_flag(df, datetime_column, sunrise_col, sunset_col):
    """Create flags only when both supplied sunrise/sunset fields parse reliably."""
    if not sunrise_col or not sunset_col or sunrise_col not in df or sunset_col not in df: return df,"Classification jour/nuit impossible directement à partir des colonnes fournies."
    def combine(values):
        text=values.astype(str).str.strip(); return pd.to_datetime(df[datetime_column].dt.date.astype(str)+" "+text,errors="coerce")
    sunrise,sunset=combine(df[sunrise_col]),combine(df[sunset_col])
    if sunrise.notna().mean()<0.9 or sunset.notna().mean()<0.9: return df,"Classification jour/nuit impossible directement à partir des colonnes fournies."
    out=df.copy(); out["is_daylight"]=(out[datetime_column]>=sunrise)&(out[datetime_column]<=sunset)
    return out,"Daylight classification derived from supplied sunrise/sunset columns."
