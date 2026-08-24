"""Availability audit for modern forecasting libraries; no misleading GFM label for one series."""
import importlib.util

def run_gfm_models(*args,**kwargs):
    """Report guarded availability; deep/GFM training requires explicitly installed backend and adequate series."""
    candidates={"N-BEATS/N-HiTS":"neuralforecast","TFT":"pytorch_forecasting","PatchTST":"darts","TimesFM":"timesfm"}
    return {},{name:("available but not auto-trained: configure a backend-specific experiment; a single local series is not a global training corpus" if importlib.util.find_spec(mod) else f"skipped: optional dependency '{mod}' is not installed") for name,mod in candidates.items()}
