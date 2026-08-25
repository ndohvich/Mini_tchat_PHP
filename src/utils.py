"""Filesystem, logging, reproducibility, and serialization helpers."""
import json, logging, os, platform, random, sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import numpy as np

def ensure_directories():
    """Create declared output directories without altering raw data."""
    for p in ["data/interim","data/processed","models","logs","results/figures","results/tables","results/predictions","results/diagnostics","results/reports"]: Path(p).mkdir(parents=True,exist_ok=True)

def setup_logging():
    """Configure a persistent pipeline log."""
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",handlers=[logging.FileHandler("logs/pipeline.log",encoding="utf-8"),logging.StreamHandler()])
    return logging.getLogger("solar_pipeline")

def set_global_seed(seed):
    """Seed supported RNGs; deterministic PyTorch settings reduce run variation."""
    os.environ["PYTHONHASHSEED"]=str(seed); random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    except ImportError: pass

def write_json(path,obj):
    """Write JSON with safe conversion for NumPy and path values."""
    def conv(v):
        if isinstance(v,(np.integer,np.floating)): return v.item()
        if isinstance(v,Path): return str(v)
        return str(v)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,default=conv),encoding="utf-8")

def save_environment(config):
    """Persist execution environment and full configuration for replication."""
    packages={n: (version(n) if _installed(n) else None) for n in ["pandas","numpy","scikit-learn","statsmodels","catboost","xgboost","lightgbm","pvlib","torch","neuralforecast","darts"]}
    write_json("results/reports/environment.json",{"timestamp_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"platform":platform.platform(),"packages":packages,"config":config,"seed":config["random"]["seed"]})
def _installed(n):
    try: version(n); return True
    except PackageNotFoundError: return False
