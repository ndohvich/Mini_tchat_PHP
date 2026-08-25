"""Tests for input loading and explicit column resolution."""
import pandas as pd
from src.data_loading import load_dataset,detect_column
from src.preprocessing import prepare_dataframe

def test_csv_load_and_sort(tmp_path):
    p=tmp_path/'a.csv';pd.DataFrame({'timestamp':['2024-01-02','2024-01-01'],'Radiation':[2,1]}).to_csv(p,index=False)
    df=load_dataset(p);out,_=prepare_dataframe(df,'timestamp',{})
    assert out.timestamp.is_monotonic_increasing

def test_detect_unique_target():
    assert detect_column(['GHI'],None,['Radiation','GHI'],'target')[0]=='GHI'
