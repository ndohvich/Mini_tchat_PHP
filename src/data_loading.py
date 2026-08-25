"""Dataset ingestion and explicit, non-arbitrary column detection."""
from pathlib import Path
import pandas as pd

def load_dataset(path, sheet_name=None):
    """Read CSV/XLS/XLSX, reporting sheets when Excel selection is ambiguous."""
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Dataset not found: {p}")
    try:
        if p.suffix.lower() in {".xlsx",".xls"}:
            sheets=pd.ExcelFile(p).sheet_names
            if sheet_name is None and len(sheets)>1: raise ValueError(f"Excel sheets available: {sheets}. Set data.sheet_name explicitly.")
            return pd.read_excel(p,sheet_name=sheet_name or sheets[0])
        if p.suffix.lower()==".csv": return pd.read_csv(p)
    except (OSError,ValueError,ImportError) as exc: raise RuntimeError(f"Could not read {p}: {exc}") from exc
    raise ValueError(f"Unsupported file extension: {p.suffix}")

def detect_column(columns, configured, candidates, role):
    """Return explicit/configured column; ambiguity is a hard scientific stop."""
    if configured:
        if configured not in columns: raise ValueError(f"Configured {role} column '{configured}' is absent.")
        return configured, "configured"
    norm={str(c).casefold():c for c in columns}; matches=[norm[c.casefold()] for c in candidates if c.casefold() in norm]
    matches=list(dict.fromkeys(matches))
    if len(matches)==1: return matches[0], "auto-detected"
    if not matches: raise ValueError(f"No {role} column detected; set data.{role}_column explicitly.")
    raise ValueError(f"Ambiguous {role} columns {matches}; set data.{role}_column explicitly.")

def dataset_inventory(df):
    """Return factual intake summary without cleaning the source data."""
    return {"rows":len(df),"columns":len(df.columns),"column_names":list(map(str,df.columns)),"dtypes":{str(k):str(v) for k,v in df.dtypes.items()},"missing":df.isna().sum().to_dict(),"duplicate_rows":int(df.duplicated().sum()),"preview":df.head().to_dict(orient="records")}
