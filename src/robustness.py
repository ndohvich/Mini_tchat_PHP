"""Month and daylight robustness summaries on locked predictions."""
import pandas as pd
from .metrics import calculate_metrics

def robustness_by_month(y,p,timestamps,denominator):
    """Report only month groups with enough observed rows for interpretable metrics."""
    d=pd.DataFrame({"y":y,"p":p,"time":pd.to_datetime(timestamps)});rows=[]
    for month,g in d.groupby(d.time.dt.to_period("M")):
        if len(g)>=2: rows.append({"Period":str(month),**calculate_metrics(g.y,g.p,denominator)})
    return pd.DataFrame(rows)
