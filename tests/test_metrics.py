"""Tests for transparent regression metric calculations."""
import numpy as np
from src.metrics import calculate_metrics

def test_perfect_prediction_metrics():
    m=calculate_metrics([0,1,2],[0,1,2]);assert m['RMSE']==0 and m['MAE']==0 and m['mape_excluded_zero_count']==1
