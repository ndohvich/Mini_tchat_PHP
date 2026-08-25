"""Configuration loading and validation for the experiment."""
from pathlib import Path
import yaml

def load_config(path="configs/experiment.yaml"):
    """Load YAML configuration, rejecting absent configuration files."""
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Configuration file not found: {p}")
    with p.open(encoding="utf-8") as f: cfg=yaml.safe_load(f) or {}
    if not cfg.get("data",{}).get("path"):
        raise ValueError("data.path is required. Put the dataset in data/raw and set its path in configs/experiment.yaml.")
    return cfg
